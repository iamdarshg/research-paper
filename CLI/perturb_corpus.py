#!/usr/bin/env python3
# Perturb corpus with aerodynamic shape modifications.
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
from typing import Iterable, Iterator
import numpy as np
import torch
from scipy.ndimage import binary_dilation, label as scipy_label
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aircraft_validity import canonicalize_aircraft_voxels

TRANSFORMS = ("tail_widen_30","tail_widen_50","wing_dihedral_up","wing_dihedral_down","nose_thin","airfoil_thicken")
PERTURBATION_GENERATOR_VERSION = "perturbation-transform-v1"
MIN_OCC = 0.005
MIN_SYMMETRY = 0.25
MIN_SPAN = 0.03
MIN_LENGTH = 0.08
MIN_COMPONENT_FRAC = 0.5


def _find_regions(v):
    occ = v > 0.5;
    x_idx = np.nonzero(np.any(occ, axis=(0,1)))[0];
    y_idx = np.nonzero(np.any(occ, axis=(0,2)))[0];
    if len(x_idx)<4 or len(y_idx)<4: return {};
    return {"x_min":int(x_idx[0]),"x_max":int(x_idx[-1]),"y_min":int(y_idx[0]),"y_max":int(y_idx[-1])}


def _scale_z_band(v, reg, x_fs, x_fe, s):
    out=v.copy(); rz,ry,rx=v.shape; xl,xh=reg["x_min"],reg["x_max"]; xr=xh-xl+1;
    sx,ex=int(xl+x_fs*xr),int(xl+x_fe*xr)+1;
    for xi in range(max(sx,0),min(ex,rx)):
        col=out[:,:,xi]; oyz=np.nonzero(col>0.5);
        if len(oyz[0])==0: continue;
        zc=(oyz[0].min()+oyz[0].max())/2.0; nc=np.zeros_like(col);
        for zi,yi in zip(*oyz): nz=max(0,min(int(round(zc+(zi-zc)*s)),rz-1)); nc[nz,yi]=1;
        out[:,:,xi]=nc;
    return out


def _dihedral(v, reg, d):
    out=v.copy(); rz,ry,rx=v.shape; yc=(reg["y_min"]+reg["y_max"])/2.0;
    hs=(reg["y_max"]-reg["y_min"])/2.0; dz=0.65; mxs=max(2,int(rz*0.04));
    for yi in range(ry):
        frac=abs(yi-yc)/max(hs,1.0);
        if frac<=dz: continue;
        t=(frac-dz)/(1.0-dz); shift=int(round(d*mxs*t));
        if shift==0: continue;
        for xi in range(rx):
            col=out[:,yi,xi]; oz=np.nonzero(col>0.5)[0];
            if len(oz)==0: continue;
            nc=np.zeros_like(col);
            for zi in oz: nz=max(0,min(int(zi)+shift,rz-1)); nc[nz]=1;
            out[:,yi,xi]=nc;
    return out


def _thicken_z(v):
    st=np.zeros((3,1,1),dtype=bool); st[0,0,0]=True; st[2,0,0]=True;
    return binary_dilation(v>0.5,structure=st).astype(np.uint8)


def apply_transform(v, tf):
    r=_find_regions(v)
    if not r: return v
    if tf=="tail_widen_30": return _scale_z_band(v,r,0.75,1.0,1.3)
    if tf=="tail_widen_50": return _scale_z_band(v,r,0.75,1.0,1.5)
    if tf=="wing_dihedral_up": return _dihedral(v,r,1)
    if tf=="wing_dihedral_down": return _dihedral(v,r,-1)
    if tf=="nose_thin": return _scale_z_band(v,r,0.0,0.15,0.7)
    if tf=="airfoil_thicken": return _thicken_z(v)
    raise ValueError("Unknown: "+tf)


def canonicalize_voxels(v: np.ndarray) -> np.ndarray:
    """Return a detached binary uint8 array of a transformed candidate."""
    return np.ascontiguousarray((np.asarray(v) > 0.5).astype(np.uint8, copy=False))


def canonical_content_hash(v: np.ndarray) -> str:
    return hashlib.sha256(canonicalize_voxels(v).tobytes()).hexdigest()


def iter_transform_candidates(v: np.ndarray, transforms: Iterable[str]) -> Iterator[tuple[str, np.ndarray, str]]:
    """Yield deterministic transformed arrays in the caller-provided order."""
    for transform in transforms:
        transformed = canonicalize_voxels(apply_transform(v, transform))
        yield transform, transformed, canonical_content_hash(transformed)


def chash(v): return canonical_content_hash(v)


def validate(v):
    try:
        t=torch.from_numpy((v>0.5).astype(np.float32))
        _,c=canonicalize_aircraft_voxels(t); m=c.get("metrics",{})
        if float(m.get("occupancy_ratio",0))<MIN_OCC: return False
        if float(m.get("symmetry_score",0))<MIN_SYMMETRY: return False
        if float(m.get("span_fraction_y",0))<MIN_SPAN: return False
        if float(m.get("length_fraction_x",0))<MIN_LENGTH: return False
        lab,n=scipy_label(v>0.5)
        if n==0: return False
        sz=np.bincount(lab.ravel())[1:]
        if sz.max()/max(sz.sum(),1)<MIN_COMPONENT_FRAC: return False
        return True
    except Exception: return False


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--output-dir",required=True)
    ap.add_argument("--transforms",default=",".join(TRANSFORMS))
    a=ap.parse_args()
    od=Path(a.output_dir); od.mkdir(parents=True,exist_ok=True); (od/"voxels").mkdir(exist_ok=True)
    tfs=[t.strip() for t in a.transforms.split(",") if t.strip()]
    src=[]
    with open(a.manifest,encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if ln: src.append(json.loads(ln))
    seen=set()
    for r in src:
        h=r.get("canonical_content_sha256") or r.get("voxel_sha256")
        if h: seen.add(h)
    out=[]
    stats={"source":len(src),"candidates":0,"accepted":0,"rejected_invalid":0,"rejected_duplicate":0}
    per_t={t:{"ok":0,"no":0} for t in tfs}
    bd=Path(a.manifest).parent
    for idx,rec in enumerate(src):
        sid=rec.get("source_id","unk_"+str(idx)); gp=rec.get("geometry_path")
        if not gp: continue
        p=Path(gp);
        if not p.is_absolute(): p=bd/p
        if not p.exists(): continue
        vox=np.load(str(p));
        if vox.ndim!=3: continue
        for tf in tfs:
            stats["candidates"]+=1
            tv=canonicalize_voxels(apply_transform(vox,tf)); c=canonical_content_hash(tv)
            if c in seen: stats["rejected_duplicate"]+=1; per_t[tf]["no"]+=1; continue
            if not validate(tv): stats["rejected_invalid"]+=1; per_t[tf]["no"]+=1; continue
            vid="perturb:"+tf+":"+sid; fn=vid.replace(":","_").replace("/","_")+".npy"
            np.save(str(od/"voxels"/fn),tv, allow_pickle=False)
            out.append({"source_id":vid,"source_type":"perturbation_expanded","parent_source_id":sid,"transform":tf,"canonical_content_sha256":c,"voxel_sha256":c,"geometry_path":"voxels/"+fn,"conditioning_mode":rec.get("conditioning_mode","unconditioned_source_metadata_only"),"split":"train","provenance":{"parent_manifest_name":Path(a.manifest).name,"transform":tf,"generator_version":PERTURBATION_GENERATOR_VERSION,"description":"Aerodynamic shape perturbation"}})
            seen.add(c); stats["accepted"]+=1; per_t[tf]["ok"]+=1
    stats["per_transform"]=per_t; stats["expanded_total"]=len(out)
    stats["generator_version"] = PERTURBATION_GENERATOR_VERSION
    stats["claim_boundary"]="Perturbed variants are shape-modified versions of validated parents, NOT independent aircraft."
    mp=od/"manifest.jsonl"; tmp=mp.with_suffix(".tmp")
    with tmp.open("w",encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r,sort_keys=True,ensure_ascii=True)+"\n")
    os.replace(tmp,mp)
    rp=od/"report.json"; rp.write_text(json.dumps(stats,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(stats,indent=2,sort_keys=True))
    return 0


if __name__=="__main__": raise SystemExit(main())

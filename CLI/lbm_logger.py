import logging
import os
from datetime import datetime

class LBMLogger:
    def __init__(self, log_file="lbm_debug.log"):
        self.logger = logging.getLogger("LBMSolver")
        self.logger.setLevel(logging.DEBUG)

        # Avoid duplicate handlers
        if not self.logger.handlers:
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)

            self.logger.addHandler(fh)

    def log_init(self, resolution, mach, reynolds, tau):
        self.logger.info(f"--- Solver Initialization ---")
        self.logger.info(f"Resolution: {resolution}^3")
        self.logger.info(f"Mach Number: {mach}")
        self.logger.info(f"Reynolds Number: {reynolds}")
        self.logger.info(f"Relaxation time (tau): {tau}")

    def log_step(self, step, fx, fz, stability, max_v):
        if step % 100 == 0:
            self.logger.debug(f"Step {step}: Fx={fx:.4e}, Fz={fz:.4e}, Stability={stability:.4e}, MaxV={max_v:.4e}")

    def log_error(self, message):
        self.logger.error(message)

    def log_info(self, message):
        self.logger.info(message)

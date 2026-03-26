import numpy as np


class Transform:
    """
    Transform class for coordinate transformation between PnP and VIO frames.
    Handles rotation (yaw only) and translation.
    VIO frame is defined relative to PnP frame.
    """
    
    def __init__(self, vio_yaw_rel_pnp=0.0, vio_translation_rel_pnp=None):
        """
        Initialize transformation parameters.
        VIO frame pose relative to PnP frame.
        
        Args:
            vio_yaw_rel_pnp: Yaw angle of VIO frame relative to PnP frame (radians)
            vio_translation_rel_pnp: Translation of VIO frame origin relative to PnP frame [x, y, z]
        """
        self.vio_yaw_rel_pnp = vio_yaw_rel_pnp
        self.vio_translation_rel_pnp = np.array(vio_translation_rel_pnp) if vio_translation_rel_pnp is not None else np.array([0.0, 0.0, 0.0])
        
        self._compute_rotation_matrices()
    
    def _compute_rotation_matrices(self):
        """Compute rotation matrices for yaw angle"""
        # Rotation matrix from VIO to PnP (yaw only)
        cos_y = np.cos(self.vio_yaw_rel_pnp)
        sin_y = np.sin(self.vio_yaw_rel_pnp)
        self.R_vio_to_pnp = np.array([
            [cos_y, -sin_y, 0],
            [sin_y, cos_y, 0],
            [0, 0, 1]
        ])
        
        # Rotation matrix from PnP to VIO (inverse)
        self.R_pnp_to_vio = self.R_vio_to_pnp.T
    
    def set_vio_frame(self, yaw, translation=None):
        """Update VIO frame parameters relative to PnP"""
        self.vio_yaw_rel_pnp = yaw
        if translation is not None:
            self.vio_translation_rel_pnp = np.array(translation)
        self._compute_rotation_matrices()
    
    def pnp_to_vio(self, pnp_position):
        """
        Transform position from PnP frame to VIO frame.
        
        Args:
            pnp_position: Position vector in PnP frame [x, y, z]
            
        Returns:
            Position vector in VIO frame [x, y, z]
        """
        pnp_pos = np.array(pnp_position)
        
        # Step 1: Remove VIO origin translation (expressed in PnP frame)
        pos_relative = pnp_pos - self.vio_translation_rel_pnp
        
        # Step 2: Rotate from PnP frame to VIO frame
        pos_vio = self.R_pnp_to_vio @ pos_relative
        
        return pos_vio
    
    def vio_to_pnp(self, vio_position):
        """
        Transform position from VIO frame to PnP frame.
        
        Args:
            vio_position: Position vector in VIO frame [x, y, z]
            
        Returns:
            Position vector in PnP frame [x, y, z]
        """
        vio_pos = np.array(vio_position)
        
        # Step 1: Rotate from VIO frame to PnP frame
        pos_pnp_relative = self.R_vio_to_pnp @ vio_pos
        
        # Step 2: Add VIO origin translation (expressed in PnP frame)
        pos_pnp = pos_pnp_relative + self.vio_translation_rel_pnp
        
        return pos_pnp
    
    def __call__(self, pnp_position):
        """
        Transform PnP position to VIO frame (alias for pnp_to_vio).
        
        Args:
            pnp_position: Position vector in PnP frame [x, y, z]
            
        Returns:
            Position vector in VIO frame [x, y, z]
        """
        return self.pnp_to_vio(pnp_position)


def rotation_matrix_yaw(yaw):
    """
    Create a rotation matrix for yaw angle only.
    
    Args:
        yaw: Yaw angle in radians
        
    Returns:
        3x3 rotation matrix
    """
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    return np.array([
        [cos_y, -sin_y, 0],
        [sin_y, cos_y, 0],
        [0, 0, 1]
    ])


def apply_rotation(position, yaw):
    """
    Apply yaw rotation to a position vector.
    
    Args:
        position: Position vector [x, y, z]
        yaw: Yaw angle in radians
        
    Returns:
        Rotated position vector
    """
    R = rotation_matrix_yaw(yaw)
    return R @ np.array(position)


def apply_translation(position, translation):
    """
    Apply translation to a position vector.
    
    Args:
        position: Position vector [x, y, z]
        translation: Translation vector [x, y, z]
        
    Returns:
        Translated position vector
    """
    return np.array(position) + np.array(translation)
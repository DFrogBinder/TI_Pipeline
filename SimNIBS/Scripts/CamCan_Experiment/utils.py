import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def view3d(
    vol: np.ndarray,
    *, 
    spacing=None,          # (sx, sy, sz) in world units; used only for surface mode
    cmap="gray",
    vmin=None, vmax=None,
    is_mask=False,         # if True, uses [0,1] display and nearest interp
    as_surface=False,      # if True, try a 3D isosurface via pyvista
    level=None,            # isosurface level; default 0.5 for masks, else mean
    title=None
):
    """
    Tri-planar 3-slice viewer for any 3D NumPy array, with optional 3D isosurface.

    vol: 3D ndarray (Z/Y/X order agnostic; uses vol.shape indexing directly)
    spacing: tuple of voxel sizes (sx, sy, sz) for surface rendering
    is_mask: treat data as binary mask for display (vmin=0, vmax=1, nearest interp)
    as_surface: requires scikit-image and pyvista; falls back to slices if unavailable
    level: marching-cubes isovalue; defaults to 0.5 for masks, else vol.mean()
    """
    assert vol.ndim == 3, "vol must be a 3D array"

    # Optional 3D surface
    if as_surface:
        try:
            from skimage.measure import marching_cubes
            import pyvista as pv
        except Exception:
            print("Surface mode needs scikit-image and pyvista; showing slices instead.")
            as_surface = False
        else:
            data = np.asarray(vol, dtype=float)
            if level is None:
                level = 0.5 if is_mask else float(np.nanmean(data))
            if spacing is None:
                spacing = (1.0, 1.0, 1.0)
            verts, faces, _, _ = marching_cubes(data, level=level, spacing=spacing)
            faces_pv = np.c_[np.full(len(faces), 3, dtype=np.int32), faces.astype(np.int32)].ravel()
            mesh = pv.PolyData(verts, faces_pv)
            pl = pv.Plotter()
            pl.add_mesh(mesh, opacity=0.6)
            pl.add_axes(); pl.show_grid()
            if title: pl.add_text(title, font_size=12)
            pl.show()
            return

    # --- Slice viewer (no extra deps) ---
    vol = np.asarray(vol)
    Z, Y, X = vol.shape  # just names for readability
    i, j, k = Z // 2, Y // 2, X // 2

    if is_mask:
        vmin, vmax = 0, 1
        interp = "nearest"
    else:
        interp = "none"

    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    if title:
        fig.suptitle(title)

    im_sag = ax[0].imshow(vol[i, :, :].T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interp)
    ax[0].set_title(f"Sagittal (z={i})"); ax[0].axis("off")

    im_cor = ax[1].imshow(vol[:, j, :].T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interp)
    ax[1].set_title(f"Coronal (y={j})"); ax[1].axis("off")

    im_axi = ax[2].imshow(vol[:, :, k].T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interp)
    ax[2].set_title(f"Axial (x={k})"); ax[2].axis("off")

    plt.tight_layout(rect=[0, 0.10, 1, 0.95])  # leave room for sliders

    # Sliders
    axcolor = 'lightgoldenrodyellow'
    ax_i = plt.axes([0.10, 0.04, 0.80, 0.02], facecolor=axcolor)
    ax_j = plt.axes([0.10, 0.02, 0.80, 0.02], facecolor=axcolor)
    ax_k = plt.axes([0.10, 0.00, 0.80, 0.02], facecolor=axcolor)

    si = Slider(ax_i, 'z', 0, Z - 1, valinit=i, valstep=1)
    sj = Slider(ax_j, 'y', 0, Y - 1, valinit=j, valstep=1)
    sk = Slider(ax_k, 'x', 0, X - 1, valinit=k, valstep=1)

    def update(_):
        zi, yj, xk = int(si.val), int(sj.val), int(sk.val)
        im_sag.set_data(vol[zi, :, :].T);      ax[0].set_title(f"Sagittal (z={zi})")
        im_cor.set_data(vol[:, yj, :].T);      ax[1].set_title(f"Coronal (y={yj})")
        im_axi.set_data(vol[:, :, xk].T);      ax[2].set_title(f"Axial (x={xk})")
        fig.canvas.draw_idle()

    si.on_changed(update); sj.on_changed(update); sk.on_changed(update)

    # Scroll to move axial by default
    def on_scroll(event):
        nonlocal k
        step = 1 if event.button == 'up' else -1
        k = np.clip(k + step, 0, X - 1)
        sk.set_val(int(k))

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    plt.show()

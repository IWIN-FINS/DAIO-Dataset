"""Atomic PNG export avoids Windows errors when a viewer holds the old file."""
from reliable_io import _temporary,_replace


def save_png(fig,path,dpi):
    temporary=_temporary(path)
    try:
        fig.savefig(temporary,format='png',dpi=dpi,facecolor='white')
        _replace(temporary,path,10)
    finally:temporary.unlink(missing_ok=True)

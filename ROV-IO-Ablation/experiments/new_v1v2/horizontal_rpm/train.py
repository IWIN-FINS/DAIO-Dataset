from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"src"))
from rovio_ablation import cli
cli.DEFAULT_DATA_ROOT=ROOT.parent/"new-dataset"
cli.DEFAULT_MANIFEST=cli.DEFAULT_DATA_ROOT/"processed/split.yaml"
if __name__=="__main__": cli.main(Path(__file__).resolve().parent)

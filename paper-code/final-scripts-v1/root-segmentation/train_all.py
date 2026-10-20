"""Run the final three-model training configuration."""
import argparse,json,shlex,subprocess,sys
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--tf-python',required=True,help='Python3.5 executable/launcher with TensorFlow1.12');a=p.parse_args();base=Path(__file__).resolve().parent;cfg=json.loads((base/'configuration.json').read_text())
    for m in ['unet','deeplab','segformer']:
        cmd=shlex.split(a.tf_python) if m=='deeplab' else [sys.executable];cmd+=[str(base/('train_tf.py' if m=='deeplab' else 'train_torch.py')),'--run',str(a.run),'--epochs',str(cfg['epochs']),'--threads',str(cfg['threads'][m]),'--batch',str(cfg['batch_size'])]
        cmd+=['--vendor',str(base/'vendor/deeplab')] if m=='deeplab' else ['--model',m]
        subprocess.run(cmd,check=True)
if __name__=='__main__':main()

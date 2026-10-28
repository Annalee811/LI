"""Add the completed same-reference20 control without changing any prediction."""
import csv,json
from build import OUT,RUN,js,csvout

def main():
    control=json.loads((RUN/'unet_control/completion.json').read_text());assert control['epochs_completed']==100
    path=OUT/'UNet_before_after_latest20_metrics.csv'
    with path.open(encoding='utf-8-sig') as f:rows=[r for r in csv.DictReader(f) if r['variant']!='same_label20_control']
    for r in control['test']['classes']:rows.append(dict(model='U-Net',variant='same_label20_control',reference='Latest20; same supervision and structured decoder without ConvLSTM/GAT',**r))
    csvout(path,rows);metrics=json.loads((OUT/'metrics.json').read_text());metrics['same_label20_control']=control['test'];js(OUT/'metrics.json',metrics)
    report=OUT/'RESULTS.md';s=report.read_text();s=s.split('\n## Same-reference control')[0]
    s+='''\n## Same-reference control\n\nThe completed100-epoch reference20 control, with the same supervision and structured\ndecoder but without ConvLSTM/GAT, gives persistent/new/tip Dice .9682/.8503/.4129,\nmacro .7438. The joint model gives .9270/.7154/.5757, macro .7394. Thus the joint\nmodules increase tip Dice by .1627 but reduce new/persistent Dice, and do not improve\nthe macro score over this matched control. No individual ConvLSTM-only or GAT-only\narm was run, so the individual module contribution is not established.\n''';report.write_text(s)
if __name__=='__main__':main()

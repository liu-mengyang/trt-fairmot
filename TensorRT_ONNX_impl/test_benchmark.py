import os
import os.path as osp
import motmetrics as mm

import datasets.dataset.jde as datasets
from test_utils import eval_seq
from fairmot.tracking_utils.log import logger
from fairmot.tracking_utils.evaluation import Evaluator
from fairmot.tracking_utils.utils import mkdir_if_missing
from opts import opts

seqs_str = '''MOT17-02-FRCNN
          MOT17-04-FRCNN
          MOT17-05-FRCNN
          MOT17-09-FRCNN
          MOT17-10-FRCNN
          MOT17-11-FRCNN
          MOT17-13-FRCNN'''
seqs = [seq.strip() for seq in seqs_str.split()]

opt = opts().init()
data_type = 'mot'
show_image = False
save_images = False
save_videos = False
exp_name = "originFairMOT"
data_root = "/workspace/dataset/MOT17/train/FRCNN"
result_root = os.path.join('..', 'results', exp_name)
mkdir_if_missing(result_root)

# run tracking
accs = []
n_frame = 0
timer_avgs, timer_calls = [], []
for seq in seqs:
    print(osp.join(data_root, seq, 'img1'))
    dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
    result_filename = os.path.join(result_root, '{}.txt'.format(seq))
    output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
    meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
    frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
    nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
    n_frame += nf
    timer_avgs.append(ta)
    timer_calls.append(tc)
    # eval
    logger.info('Evaluate seq: {}'.format(seq))
    evaluator = Evaluator(data_root, seq, data_type)
    accs.append(evaluator.eval_file(result_filename))
    if save_videos:
        output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
        os.system(cmd_str)
timer_avgs = np.asarray(timer_avgs)
timer_calls = np.asarray(timer_calls)
all_time = np.dot(timer_avgs, timer_calls)
avg_time = all_time / np.sum(timer_calls)
logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

# get summary
metrics = mm.metrics.motchallenge_metrics
mh = mm.metrics.create()
summary = Evaluator.get_summary(accs, seqs, metrics)
strsummary = mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names
)
print(strsummary)
Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))
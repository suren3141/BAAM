import os
import argparse

# import some common detectron2 utilities
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import launch

# import custom code
import model
from trainner import train_model
from tester import test_model
from model.config import add_default_config



def main(cfg):

    # Prepare save folder
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.system(f'cp {config} {cfg.OUTPUT_DIR}/config.yaml')

    # Load model and pre-trained weights
    print('Load.... model')
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    print('Done...')

    # Train
    if args.train:
        print('#####################')
        print('##                 ##')
        print('##   Train model   ##')
        print('##                 ##')
        print('#####################')
        train_model(cfg, model)

    # Test
    test_model(cfg, model)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', dest='train', action='store_true', help='train model before evaluation')
    parser.add_argument('-c', '--config', type=str, default = 'configs/custom.yaml', help='config file')
    parser.add_argument('-o', '--output', type=str, help='output directory')
    args = parser.parse_args()

    print(args)


    # Load config
    setup_logger()
    cfg = get_cfg()
    add_default_config(cfg)
    config = args.config
    #config = 'configs/custom.yaml'
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config)

    if args.output is not None:
        cfg.OUTPUT_DIR = args.output
    print(cfg)
    main(cfg)

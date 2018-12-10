from peleenet_ssd import build_ssd
import config as cfg
from train import train
from test import test


# from test import test


def main():
    model = build_ssd(cfg)

    train(cfg, model)

    # test(cfg, model)


if __name__ == '__main__':
    main()

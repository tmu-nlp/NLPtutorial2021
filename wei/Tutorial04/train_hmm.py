
if __name__ == '__main__':
    import os, sys, pprint
    sys.path.append(os.path.pardir)
    from utils.pos_model import load_data, PosModel
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_file', required=True, type=str)
    parser.add_argument('-m', '--model_file', required=True, type=str)
    arg = parser.parse_args()
    # train_file='../data/wiki-en-train.norm_pos'    --モデル学習用のラベルありテータ
    # model_file='model_file.pkl'                   　――モデルパラメータ保存
    data = load_data(arg.train_file, decorator=lambda w: w.lower())
    data = data[:1]          # for viewing the outputs

    model = PosModel()
    model.train(data)

    model.save_params(arg.model_file)
    model.load_params(arg.model_file)       # for viewing the model params

    print(f'model file saved to {arg.model_file}')


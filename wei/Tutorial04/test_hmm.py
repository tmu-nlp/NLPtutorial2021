
if __name__ == '__main__':
    import os, sys, pprint
    sys.path.append(os.path.pardir)
    from utils.pos_model import load_data, PosModel
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-te', '--test_file', required=True, type=str)
    parser.add_argument('-m', '--model_file', required=True, type=str)
    arg = parser.parse_args()
    # test_file='../test/05-train-input.txt'　    --モデルを使って品詞推定を行う
    # model_file='model_file.pkl'           　　　――学習済みモデルパラメータ

    test_data = load_data(arg.test_file, mode='test', decorator=lambda w: w.lower())
    # print(test_data)

    model = PosModel()
    model.load_params(arg.model_file)
    estimate = model.predict_pos(test_data)

    for line in estimate:
        print(' '.join(line))


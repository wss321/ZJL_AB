if __name__ == '__main__':
    print(' *************** 0/7.dataset_utils ***************  ')
    from dataset_utils import dataset_utils_main

    dataset_utils_main()
    print(' *************** 1/7.create_pickle_file ***************  ')
    from create_pickle_file import create_pickle_file_main

    create_pickle_file_main()
    print(' *************** 2/7.read_zjl *************** ')
    from read_zjl import read_zjl_main

    read_zjl_main()
    print(' *************** 3/7.keras_train *************** ')
    from keras_train import keras_train_main

    keras_train_main()
    print(' *************** 4/7.create_train_visual_feature *************** ')
    from create_train_visual_feature import create_train_visual_feature_main

    create_train_visual_feature_main()
    print(' *************** 5/7.train_DEM *************** ')
    from train_DEM import train_dem_main

    train_dem_main(epoches=1000)
    print(' *************** 6/7.create_test_visual_feature *************** ')
    from create_test_visual_feature import create_test_visual_feature_main

    create_test_visual_feature_main()
    print(' *************** 7/7.DEM_predict *************** ')
    from DEM_predict import dem_predict_main

    dem_predict_main()
    print(' *************** ALL DONE. *************** .')

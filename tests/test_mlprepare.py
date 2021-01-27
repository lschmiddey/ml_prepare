from prep_funcs import *
import unittest

class TestCases(unittest.TestCase):
    def test_ifnone(self):
        a, b = None, 5
        expected_result = 5
        self.assertEqual(ifnone(a,b), expected_result)
        
        a,b = 10,5
        expected_result = 10
        self.assertEqual(ifnone(a,b), expected_result)
        
    def test_df_to_type(self):
        
        data = {'PassengerId': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
                 'Survived': {0: 0, 1: 1, 2: 1, 3: 1, 4: 0},
                 'Sex': {0: 'male', 1: 'female', 2: 'female', 3: 'female', 4: 'male'},
                 'Age': {0: 22.0, 1: 38.0, 2: 26.0, 3: 35.0, 4: 35.0},
                 'Cabin': {0: np.NaN, 1: 'C85', 2: np.NaN, 3: 'C123', 4: np.NaN},
                 'Fake_date': {0: '1995-04-01T00:00:00.000000000',
                  1: '1998-10-27T00:00:00.000000000',
                  2: '1997-03-05T00:00:00.000000000',
                  3: '1999-11-30T00:00:00.000000000',
                  4: '1994-02-01T00:00:00.000000000'}}

        df = pd.DataFrame(data)
        
        date_type = ['Fake_date']
        continuous_type = ['Age', 'PassengerId']
        categorical_type = ['Sex', 'Cabin', 'Survived']
        
        expected_rows = 5
        expected_cols = 18
        
        ml_instance = MLPrepare()
        test_result = ml_instance.df_to_type(df, date_type, continuous_type, categorical_type)
        
        self.assertEqual(test_result.shape[0], expected_rows)
        self.assertEqual(test_result.shape[1], expected_cols)

    def test_split_df(self):
        
        data = {'PassengerId': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
                 'Survived': {0: 0, 1: 1, 2: 1, 3: 1, 4: 0},
                 'Sex': {0: 'male', 1: 'female', 2: 'female', 3: 'female', 4: 'male'},
                 'Fake_Week': {0: 2, 1: 46, 2: 34, 3: 12, 4: 46},
                 'Age': {0: 22.0, 1: 38.0, 2: 26.0, 3: 35.0, 4: 35.0},
                 'Cabin': {0: np.NaN, 1: 'C85', 2: np.NaN, 3: 'C123', 4: np.NaN},
                 'Fake_Year': {0: 1985, 1: 1990, 2: 1986, 3: 1999, 4: 1986},
                 'Fake_Month': {0: 1, 1: 11, 2: 8, 3: 3, 4: 11},
                 'Fake_Day': {0: 8, 1: 13, 2: 22, 3: 24, 4: 14},
                 'Fake_Dayofweek': {0: 1, 1: 1, 2: 4, 3: 2, 4: 4},
                 'Fake_Dayofyear': {0: 8, 1: 317, 2: 234, 3: 83, 4: 318},
                 'Fake_Is_month_end': {0: False, 1: False, 2: False, 3: False, 4: False},
                 'Fake_Is_month_start': {0: False, 1: False, 2: False, 3: False, 4: False},
                 'Fake_Is_quarter_end': {0: False, 1: False, 2: False, 3: False, 4: False},
                 'Fake_Is_quarter_start': {0: False, 1: False, 2: False, 3: False, 4: False},
                 'Fake_Is_year_end': {0: False, 1: False, 2: False, 3: False, 4: False},
                 'Fake_Is_year_start': {0: False, 1: False, 2: False, 3: False, 4: False},
                 'Fake_Elapsed': {0: 473990400,
                  1: 658454400,
                  2: 525052800,
                  3: 922233600,
                  4: 532310400}}

        df = pd.DataFrame(data)
        
        dep_var = 'Survived'
        cols = ['PassengerId',
                 'Sex',
                 'Fake_Week',
                 'Age',
                 'Cabin',
                 'Fake_Year',
                 'Fake_Month',
                 'Fake_Day',
                 'Fake_Dayofweek',
                 'Fake_Dayofyear',
                 'Fake_Is_month_end',
                 'Fake_Is_month_start',
                 'Fake_Is_quarter_end',
                 'Fake_Is_quarter_start',
                 'Fake_Is_year_end',
                 'Fake_Is_year_start',
                 'Fake_Elapsed']

        exptd_X_train_shape, exptd_X_test_shape, exptd_y_train_shape, exptd_y_test_shape = (3, 17), (2, 17), (3,), (2,)
        
        ml_instance = MLPrepare()
        X_train, X_test, y_train, y_test = ml_instance.split_df(df=df, dep_var=dep_var, \
                                                                test_size=0.3, split_mode='random', split_var=None, cond=None)
        

        
        self.assertEqual(X_train.shape, exptd_X_train_shape)
        self.assertEqual(X_test.shape, exptd_X_test_shape)
        self.assertEqual(y_train.shape, exptd_y_train_shape)
        self.assertEqual(y_test.shape, exptd_y_test_shape)
        
    def test_cat_transform(self):
        
        ml_instance = MLPrepare()
        
        categorical_type = ['Sex', 'Cabin']

        traindata = {'PassengerId': {827: 828, 56: 57, 746: 747, 762: 763, 767: 768},
                 'Sex': {827: 'male', 56: 'female', 746: 'male', 762: 'male', 767: 'female'},
                 'Fake_Week': {827: 44, 56: 2, 746: 5, 762: 35, 767: 47},
                 'Age': {827: 1.0, 56: 21.0, 746: 16.0, 762: 20.0, 767: 30.5},
                 'Cabin': {827: 'CA', 56: '#NaN', 746: '#NaN', 762: '#NaN', 767: '#NaN'},
                 'Fake_Year': {827: 1980, 56: 1988, 746: 1996, 762: 1991, 767: 1981},
                 'Fake_Month': {827: 11, 56: 1, 746: 1, 762: 8, 767: 11},
                 'Fake_Day': {827: 2, 56: 15, 746: 31, 762: 31, 767: 16},
                 'Fake_Dayofweek': {827: 6, 56: 4, 746: 2, 762: 5, 767: 0},
                 'Fake_Dayofyear': {827: 307, 56: 15, 746: 31, 762: 243, 767: 320}}

        testdata = {'PassengerId': {273: 274, 825: 826, 561: 562, 640: 641, 383: 384},
                     'Sex': {273: 'male', 825: 'male', 561: 'male', 640: 'male', 383: 'female'},
                     'Fake_Week': {273: 30, 825: 15, 561: 22, 640: 51, 383: 41},
                     'Age': {273: 37.0, 825: np.NaN, 561: 40.0, 640: 20.0, 383: 35.0},
                     'Cabin': {273: 'C118', 825: 'CA', 561: '#NaN', 640: '#NaN', 383: '#NaN'},
                     'Fake_Year': {273: 1996, 825: 1980, 561: 1996, 640: 1983, 383: 1984},
                     'Fake_Month': {273: 7, 825: 4, 561: 6, 640: 12, 383: 10},
                     'Fake_Day': {273: 27, 825: 7, 561: 1, 640: 24, 383: 8},
                     'Fake_Dayofweek': {273: 5, 825: 0, 561: 5, 640: 5, 383: 0},
                     'Fake_Dayofyear': {273: 209, 825: 98, 561: 153, 640: 358, 383: 282}}

        train_df = pd.DataFrame(traindata)       
        test_df = pd.DataFrame(testdata)

        train_df[categorical_type] = train_df[categorical_type].astype('category')
        test_df[categorical_type] = test_df[categorical_type].astype('category')
        
        exptd_train_shape, exptd_test_shape, exptd_dict_len, exptd_inv_dict_len = (5, 10), (5, 10), 3, 3
        
        X_train_, X_test_, dict_list, dict_inv_list = ml_instance.cat_transform(train_df, test_df, categorical_type, path='')
        
        self.assertEqual(X_train_.shape, exptd_train_shape)
        self.assertEqual(X_test_.shape, exptd_test_shape)
        self.assertEqual(len(dict_list[0]), exptd_dict_len)
        self.assertEqual(len(dict_inv_list[0]), exptd_inv_dict_len)
        
    def test_cont_standardize(self):
        
        ml_instance = MLPrepare()
        
        categorical_type = ['Sex', 'Cabin']

        traindata = {'PassengerId': {201: 202, 420: 421, 594: 595, 531: 532, 596: 597},
         'Sex': {201: 2, 420: 2, 594: 2, 531: 2, 596: 1},
         'Fake_Week': {201: 12, 420: 8, 594: 14, 531: 10, 596: 33},
         'Age': {201: np.NaN, 420: np.NaN, 594: 37.0, 531: np.NaN, 596: 20},
         'Cabin': {201: 0, 420: 0, 594: 0, 531: 0, 596: 0}}

        testdata = {'PassengerId': {118: 119, 734: 735, 558: 559},
         'Sex': {118: 2, 734: 2, 558: 1},
         'Fake_Week': {118: 14, 734: 48, 558: 22},
         'Age': {118: 24.0, 734: 23.0, 558: np.NaN},
         'Cabin': {118: 37, 734: 0, 558: 135}}

        traindata_df = pd.DataFrame(traindata)
        testdata_df = pd.DataFrame(testdata)

        y_train = pd.Series(np.array([0, 0, 0, 0, 1, 1, 0, 0]))
        y_test = pd.Series(np.array([0, 0, 1]))
        
        exptd_train_shape, exptd_test_shape, exptd_y_train_len, exptd_y_test_len = (5, 5), (3, 5), 8, 3
        
        exptd_age_mean = 0.0
        
        X_train_2, X_test_2, y_train_2, y_test_2, scaler = ml_instance.cont_standardize(traindata_df, testdata_df, y_train, y_test, cat_type=categorical_type, transform_y=False, id_type='PassengerId', path='', standardizer='StandardScaler')
        
        Age_mean = X_train_2.Age.mean()

        self.assertEqual(X_train_2.shape, exptd_train_shape)
        self.assertEqual(X_test_2.shape, exptd_test_shape)
        self.assertEqual(len(y_train_2), exptd_y_train_len)
        self.assertEqual(len(y_test_2), exptd_y_test_len)
        self.assertEqual(Age_mean, exptd_age_mean)
        
    def test_cont_standardize_groupby(self):
        
        ml_instance = MLPrepare()
        
        id_type='PassengerId'
        cont_type = ['Age']

        traindata = {'PassengerId': {201: 202, 420: 202, 594: 202, 531: 532, 596: 532},
         'Sex': {201: 2, 420: 2, 594: 2, 531: 2, 596: 1},
         'Fake_Week': {201: 12, 420: 8, 594: 14, 531: 10, 596: 33},
         'Age': {201: 85., 420: 96., 594: 37.0, 531: 22., 596: np.NaN},
         'Cabin': {201: 0, 420: 0, 594: 0, 531: 0, 596: 0}}

        testdata = {'PassengerId': {118: 119, 734: 735, 558: 559},
         'Sex': {118: 2, 734: 2, 558: 1},
         'Fake_Week': {118: 14, 734: 48, 558: 22},
         'Age': {118: 24.0, 734: 23.0, 558: np.NaN},
         'Cabin': {118: 37, 734: 0, 558: 135}}

        traindata_df = pd.DataFrame(traindata)
        testdata_df = pd.DataFrame(testdata)
        
        exptd_df_shape = (5, 5)
        
        exptd_std_age = 0.48145422560319073
        
        df_, scaler = ml_instance.cont_standardize_groupby(df=traindata_df, cont_type=cont_type, id_type=id_type, path='', standardizer='StandardScaler')
        
        self.assertEqual(df_.shape, exptd_df_shape)
        self.assertAlmostEqual(df_.loc[201,'Age'], exptd_std_age)
        
        
        
        
if __name__ == '__main__':
    unittest.main()
    
    
    
    
    
    
    
    
    
    
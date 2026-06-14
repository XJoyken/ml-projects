import pandas as pd

def cabin_embarked_transformer(df: pd.DataFrame) -> pd.DataFrame:
    df['Cabin'] = df['Cabin'].fillna('U')
    df['Cabin'] = df['Cabin'].apply(lambda x: x[0])
    df['Embarked'] = df['Embarked'].fillna('U')
    return df

def title_group_creation(df: pd.DataFrame) -> pd.DataFrame:
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    group_size = df['Ticket'].value_counts()
    df['GroupSize'] = df['Ticket'].map(group_size)
    df = df.drop(columns=['Name', 'Ticket'])
    return df

def fit_age_imputation(df: pd.DataFrame) -> dict:
    median_dict = {}
    global_median = df['Age'].median()
    titles = df['Title'].unique()
    for title in titles:
        title_age = df.loc[df['Title'] == title, 'Age']

        Q1 = title_age.quantile(0.25)
        Q3 = title_age.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - IQR * 1.5
        upper_bound = Q3 + IQR * 1.5

        outlier_mask = (title_age > upper_bound) | (title_age < lower_bound)
        cleaned_title_age = title_age[~outlier_mask]
        median_title_age = cleaned_title_age.median()

        if pd.isna(median_title_age):
            median_dict[title] = global_median
        else:
            median_dict[title] = median_title_age

    median_dict['__global__'] = global_median
    return median_dict

def transform_age_imputation(df: pd.DataFrame, median_dict: dict) -> pd.DataFrame:
    df = df.copy()
    for title, median_value in median_dict.items():
        if title == '__global__':
            continue
        missing_title_age = (df['Title'] == title) & (df['Age'].isna())
        df.loc[missing_title_age, 'Age'] = median_value
    global_median = median_dict.get('__global__', df['Age'].median())
    df['Age'] = df['Age'].fillna(global_median)

    return df

def family_size_creation(df: pd.DataFrame) -> pd.DataFrame:
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df = df.drop(columns=['SibSp', 'Parch'])
    return df

def merge_test() -> pd.DataFrame:
    df_test = pd.read_csv('data/test.csv')
    df_gs = pd.read_csv('data/gender_submission.csv')
    df_merged = pd.merge(df_test, df_gs, on='PassengerId', how='inner')
    return df_merged
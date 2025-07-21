from config.database_config import DB_CONFIG
from config.paths_config import RAW_DIR, TEST_PATH, TRAIN_PATH
from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessing
from src.feature_store import RedisFeatureStore
from src.model_training import ModelTraining


if __name__=="__main__":
    data_ingestion = DataIngestion(DB_CONFIG, RAW_DIR)
    data_ingestion.run() 

    feature_store = RedisFeatureStore()

    data_processor = DataProcessing(TRAIN_PATH, TEST_PATH, feature_store)
    data_processor.run()

    feature_store = RedisFeatureStore()

    model_trainer = ModelTraining(feature_store)
    model_trainer.run()
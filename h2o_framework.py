import h2o
from h2o.automl import H2OAutoML
import pandas as pd
from sklearn.metrics import accuracy_score
from h2o.estimators import H2ODeepLearningEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator


# Inicialización del servidor H2O
h2o.init(ip='13.51.165.229', port=54321, max_mem_size="0.5G")

# Función para cargar y limpiar datos
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    return h2o.H2OFrame(df)

# Cargar datos de entrenamiento inicial (lunes y martes)
train_files = [
    "CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv",
    "CICIDS2017/Tuesday-WorkingHours.pcap_ISCX.csv"
]
train_frames = [load_and_prepare_data(file) for file in train_files]
train = train_frames[0].rbind(train_frames[1])  # Concatenar H2OFrames

# Lista de predictores y respuesta
predictors = ["Destination port", "Protocol", "Flow duration", "Total fwd packets", "Total backward packets",
    "Total length of fwd packets", "Total length of bwd packets", "Fwd packet length max", 
    "Fwd packet length min", "Fwd packet length mean", "Fwd packet length std", 
    "Bwd packet length max", "Bwd packet length min", "Bwd packet length mean", 
    "Bwd packet length std", "Flow bytes/s", "Flow packets/s", "Flow IAT mean", 
    "Flow IAT std", "Flow IAT max", "Flow IAT min", "Fwd IAT total", "Fwd IAT mean", 
    "Fwd IAT std", "Fwd IAT max", "Fwd IAT min", "Bwd IAT total", "Bwd IAT mean", 
    "Bwd IAT std", "Bwd IAT max", "Bwd IAT min", "Fwd PSH flags", "Bwd PSH flags", 
    "Fwd URG flags", "Bwd URG flags", "Fwd header length", "Bwd header length", 
    "Fwd packets/s", "Bwd packets/s", "Min packet length", "Max packet length", 
    "Packet length mean", "Packet length std", "Packet length variance", "FIN flag count", 
    "SYN flag count", "RST flag count", "PSH flag count", "ACK flag count", 
    "URG flag count", "CWE flag count", "ECE flag count", "Down/up ratio", 
    "Average packet size", "Avg fwd segment size", "Avg bwd segment size", 
    "Fwd header length", "Fwd avg bytes/bulk", "Fwd avg packets/bulk", 
    "Fwd avg bulk rate", "Bwd avg bytes/bulk", "Bwd avg packets/bulk", 
    "Bwd avg bulk rate", "Subflow fwd packets", "Subflow fwd bytes", 
    "Subflow bwd packets", "Subflow bwd bytes", "Init_Win_bytes_forward", 
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward", 
    "Active mean", "Active std", "Active max", "Active min", "Idle mean", 
    "Idle std", "Idle max", "Idle min"
]

response = "Label"

# Función para entrenar modelo H2OAutoML
def train_automl(train, valid=None, max_runtime_secs=60, checkpoint_model=None):
    #TODO: change the checkpoint philosophy, if it exists then load the checkpointed model,
    if checkpoint_model:
        #load checkpointed model from models folder
        aml = h2o.load_model(checkpoint_model)
        aml.train(x=predictors, y=response, training_frame=train, validation_frame=valid)
    else:
        aml = H2OAutoML(max_runtime_secs=max_runtime_secs, seed=1234, verbosity="info", nfolds=0, keep_cross_validation_predictions=False,
                        include_algos=['DeepLearning', 'DRF'])
        aml.train(x=predictors, y=response, training_frame=train, validation_frame=valid)
    return aml

# Entrenamiento inicial del modelo
aml = train_automl(train)

# Guardar el modelo inicial
best_model = aml.leader
best_model.model_id = "best_model"
h2o.download_model(model=best_model, path="models/")

# Días a procesar para validación y retraining
validation_days = [
    "CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv",
    "CICIDS2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "CICIDS2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
]

# Función para asegurar que los niveles categóricos coincidan
# Función para asegurar que los niveles categóricos coincidan
def ensure_categorical_levels(train, valid, response):
    for col in train.columns:
        if train[col].isfactor()[0]:
            # Convertir a factor y alinear niveles
            valid[col] = valid[col].asfactor()
            
            # Extraer los niveles como una lista de cadenas
            train_levels = train[col].levels()[0]
            valid[col] = valid[col].set_levels(train_levels)


# Ejemplo de uso en el ciclo de reentrenamiento
for day_file in validation_days:
    # Cargar datos de validación
    valid = load_and_prepare_data(day_file)
    
    # Asegurarse de que los niveles categóricos coincidan
    ensure_categorical_levels(train, valid, response)
    
    # Realizar predicciones
    predictions = best_model.predict(valid)

    # Convertir predicciones y etiquetas reales a formato de pandas para comparación
    pred_df = predictions.as_data_frame()['predict']
    actual_df = valid.as_data_frame()[response]

    accuracy = accuracy_score(actual_df, pred_df)
    print(f"Performance for {day_file}: Accuracy = {accuracy}")

    # Decidir si reentrenar
    if accuracy < 0.95:
        print(f"Retraining model for {day_file}")
        
        # Agregar datos del día al conjunto de entrenamiento
        train = train.rbind(valid)
        
        # Reentrenar modelo
        retrainable_model = h2o.upload_model("models/best_model")
        
        # Comprobar el tipo de modelo y reentrenar
        if retrainable_model.algo == "DeepLearning":
            dl_checkpoint2 = H2ODeepLearningEstimator(
                model_id="best_model_DL_" + str(retrainable_model.params['epochs']['actual'] + 1),
                checkpoint=retrainable_model.model_id,
                epochs=int(retrainable_model.params['epochs']['actual']) + 5,
                seed=retrainable_model.params['seed']['actual']
            )
        elif retrainable_model.algo == "drf":
            dl_checkpoint2 = H2ORandomForestEstimator(
                model_id="best_model_forest_" + str(retrainable_model.params['ntrees']['actual'] + 1),
                checkpoint=retrainable_model.model_id,
                ntrees=int(retrainable_model.params['ntrees']['actual']) + 5,
                seed=retrainable_model.params['seed']['actual']
            )
        
        # Entrenar con el conjunto de entrenamiento completo
        dl_checkpoint2.train(x=predictors, y=response, training_frame=train)
        
        # Guardar nuevo mejor modelo
        best_model_path = h2o.download_model(model=dl_checkpoint2, path="models/")


# Finalización del servidor H2O
h2o.cluster().shutdown(prompt=False)

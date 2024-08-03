import h2o
from h2o.automl import H2OAutoML
import pandas as pd

# Inicialización del servidor H2O
h2o.init(min_mem_size=2)

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
predictors = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
    "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std",
    "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
]
response = "Label"

# Función para entrenar modelo H2OAutoML
def train_automl(train, valid=None, max_runtime_secs=60, checkpoint_model=None):
    aml = H2OAutoML(max_runtime_secs=max_runtime_secs, seed=1234, verbosity="info",
                    project_name='IDS_project', nfolds=0, keep_cross_validation_predictions=False,
                    exclude_algos=['DeepLearning', 'StackedEnsemble'])
    aml.train(x=predictors, y=response, training_frame=train, validation_frame=valid)
    return aml

# Entrenamiento inicial del modelo
aml = train_automl(train)

# Guardar el modelo inicial
best_model = aml.leader
best_model_path = h2o.save_model(model=best_model, path="models", force=True)

# Días a procesar para validación y retraining
validation_days = [
    "CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv",
    "CICIDS2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "CICIDS2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
]

# Ciclo de validación y retraining
for day_file in validation_days:
    # Cargar datos de validación
    valid = load_and_prepare_data(day_file)

    # Validar modelo actual
    performance = best_model.model_performance(valid)
    print(f"Performance for {day_file}: {performance.auc()}")

    # Decidir si reentrenar
    if performance.auc() < 0.95:  # Umbral de ejemplo
        print(f"Retraining model for {day_file}")
        
        # Agregar datos del día al conjunto de entrenamiento
        train = train.rbind(valid)
        
        # Reentrenar modelo
        aml = train_automl(train, valid, checkpoint_model=best_model)
        
        # Obtener nuevo mejor modelo
        best_model = aml.leader
        
        # Guardar nuevo mejor modelo
        best_model_path = h2o.save_model(model=best_model, path="models", force=True)

# Finalización del servidor H2O
h2o.shutdown(prompt=False)

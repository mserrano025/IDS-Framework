# %% [markdown]
# Initial Imports

# %%
import h2o
from h2o.automl import H2OAutoML
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from h2o.estimators import H2ODeepLearningEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

# %% [markdown]
# # Server Start

# %%
h2o.init(min_mem_size='8G')

# %% [markdown]
# ## Training of the initial basic model

# %%
def load_and_prepare_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='ISO-8859-1', errors='replace', low_memory=False)
    
    df.columns = df.columns.str.strip()
    df['Label_Binary'] = df['Label'].apply(lambda x: 'BENIGN' if x == 'BENIGN' else 'ATTACK')
    
    # Convertir 'Timestamp' a datetime y ordenar
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by='Timestamp')
    
    # Forzar 'Label_Binary' como categoría
    df['Label_Binary'] = df['Label_Binary'].astype('category')
    
    h2o_frame = h2o.H2OFrame(df)

    # Asegurarse de que las columnas sean categóricas donde sea necesario
    h2o_frame['Label_Binary'] = h2o_frame['Label_Binary'].asfactor()

    return h2o_frame

def ensure_matching_types(frame1, frame2):
    # Asegurarse de que las columnas categóricas sean del mismo tipo
    for column in frame1.columns:
        if frame1.types[column] == 'enum' and frame2.types[column] != 'enum':
            frame2[column] = frame2[column].asfactor()
        elif frame2.types[column] == 'enum' and frame1.types[column] != 'enum':
            frame1[column] = frame1[column].asfactor()


day_files = {
    "Wednesday": ["CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv"],
    "Thursday": ["CICIDS2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                 "CICIDS2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"],
    "Friday": ["CICIDS2017/Friday-WorkingHours-Morning.pcap_ISCX.csv",
               "CICIDS2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
               "CICIDS2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"]
}
day_selection = input("Seleccione el día para analizar (Wednesday, Thursday, Friday): ").capitalize()
training_files = []
if day_selection == "Wednesday":
    training_files = ["CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv", "CICIDS2017/Tuesday-WorkingHours.pcap_ISCX.csv"]
elif day_selection == "Thursday":
    training_files = ["CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv", "CICIDS2017/Tuesday-WorkingHours.pcap_ISCX.csv",
                      "CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv"]
elif day_selection == "Friday":
    training_files = ["CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv", "CICIDS2017/Tuesday-WorkingHours.pcap_ISCX.csv",
                      "CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv",
                      "CICIDS2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                      "CICIDS2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"]
else:
    print("Día no válido seleccionado. Ejecución cancelada.")
    h2o.cluster().shutdown(prompt=False)
    exit()

training_frames = [load_and_prepare_data(file) for file in training_files]
training_data = training_frames[0]
for frame in training_frames[1:]:
    training_data = training_data.rbind(frame)

# Cargar y preparar datos del día seleccionado
selected_day_frames = [load_and_prepare_data(file) for file in day_files[day_selection]]
selected_day_data = selected_day_frames[0]
for frame in selected_day_frames[1:]:
    selected_day_data = selected_day_data.rbind(frame)
# %%
# Lista de predictores y respuesta
predictors = [
    #"Destination Port", 
    "Protocol","Flow Duration", "Total Fwd Packets", 
    "Total Backward Packets", "Total Length of Fwd Packets", 
    "Total Length of Bwd Packets", "Fwd Packet Length Max", 
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std", 
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", 
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", 
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", 
    "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", 
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", 
    "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", 
    "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length", 
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length", 
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance", 
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", 
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count", 
    "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", 
    "Avg Bwd Segment Size", "Fwd Header Length", "Fwd Avg Bytes/Bulk", 
    "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", 
    "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets", 
    "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes", 
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd", 
    "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", 
    "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
]

response = "Label"
response_binary = "Label_Binary"

# %%

def train_binary(train, valid=None, max_runtime_secs=60):
    aml = H2OAutoML(max_runtime_secs=max_runtime_secs, seed=1234, verbosity="info", nfolds=0, keep_cross_validation_predictions=False
                )
    aml.train(x=predictors, y=response_binary, training_frame=train, validation_frame=valid)
    new_model = aml.leader
    return new_model

# %%
def model_metrics_evaluation(model_predictions, ground_truth, label='ATTACK'):
    accuracy = accuracy_score(ground_truth.as_data_frame(), model_predictions.as_data_frame())
    f1 = f1_score(ground_truth.as_data_frame(), model_predictions.as_data_frame(), pos_label=label)
    recall = recall_score(ground_truth.as_data_frame(), model_predictions.as_data_frame(), pos_label=label)
    precision = precision_score(ground_truth.as_data_frame(), model_predictions.as_data_frame(), pos_label=label)
    confusion = confusion_matrix(ground_truth.as_data_frame(), model_predictions.as_data_frame(), labels=['BENIGN', 'ATTACK'])
    return accuracy, f1, recall, precision, confusion

# %% [markdown]
# ### Pipeline

# %%

#Binary clasification
model = train_binary(training_data)
print(model)

day_df = h2o.as_list(selected_day_data, use_pandas=True)
day_df['Timestamp'] = pd.to_datetime(day_df['Timestamp'], unit='ms')
start_time = day_df['Timestamp'].min()
end_time = day_df['Timestamp'].max()
time_ranges = pd.date_range(start=start_time, end=end_time, freq='7min30s')

day_hours = [day_df[(day_df['Timestamp'] >= time_ranges[i]) & (day_df['Timestamp'] < time_ranges[i+1])] 
             for i in range(len(time_ranges)-1)]
day_hours = [hour for hour in day_hours if len(hour) > 0]
day_hours_h2o = [h2o.H2OFrame(hour_df) for hour_df in day_hours]

# %%
# Inicializa una lista para almacenar los registros de métricas
metrics_log = []

# Pipeline
past_hours = []
for hour in day_hours_h2o:
    predictions_hour = model.predict(hour)
    print(len(predictions_hour['predict']))
    accuracy, f1, recall, precision, confusion = model_metrics_evaluation(predictions_hour['predict'], hour['Label_Binary'], label='ATTACK')

    # Guardar las métricas en la lista
    metrics_log.append({
        "timestamp": datetime.fromtimestamp(hour['Timestamp'][0, 0] / 1000.0).strftime('%H:%M:%S'),
        "algorithm": model.algo,
        "accuracy": accuracy,
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "true_positives": confusion[0][0],
        "true_negatives": confusion[1][1],
        "false_positives": confusion[0][1],
        "false_negatives": confusion[1][0],
        "retrained": False  # Se actualizará a True si se reentrena
    })
    
    past_hours.append(hour)
    
    # Check if retraining is needed
    if precision < 0.6 and confusion[1][0] > 100:
        # Concatenate the past hours
        time_passed = past_hours[0]
        for i in range(1, len(past_hours)):
            ensure_matching_types(time_passed, past_hours[i])
            time_passed = time_passed.rbind(past_hours[i])
        
        # Model accuracy on training data
        total_time = training_data.rbind(time_passed)
        new_model = train_binary(total_time)
        model = new_model
        print("Model retrained")
        
        # Actualizar el último registro de métricas para indicar que hubo reentrenamiento
        metrics_log[-1]["retrained"] = True
        
    print("---------------------------------------------------")

# Al finalizar el loop, guarda las métricas en un archivo CSV
metrics_df = pd.DataFrame(metrics_log)
metrics_csv_filename = f'model_metrics_log_{day_selection}.csv'
metrics_df.to_csv(metrics_csv_filename, index=False)

h2o.cluster().shutdown(prompt=False)

# %%

# Convertir a numpy arrays las listas involucradas
timestamps = np.array(metrics_df['timestamp'].tolist())

#TODO: esto está mal, vamos a calcularlo a partir de la matriz de confusión
attacks_received = np.array(metrics_df['true_negatives'] + metrics_df['false_negatives'])
attacks_detected = np.array(metrics_df['true_negatives'])

retrain_timestamps = np.array(metrics_df[metrics_df['retrained'] == True]['timestamp'].tolist())

# Crear el plot
plt.figure(figsize=(20, 10))

# Plot de ataques recibidos y detectados
plt.plot(timestamps, attacks_received, label='Attacks Received', color='red', marker='o')
plt.plot(timestamps, attacks_detected, label='Attacks Detected', color='green', marker='x')

# Añadir marcas donde se reentrenó el modelo
for retrain_time in retrain_timestamps:
    plt.axvline(x=retrain_time, color='blue', linestyle='--', label='Model Retrained')

plt.xlabel('Time')
plt.ylabel('Number of Attacks')
plt.title('Attacks Received vs. Attacks Detected Over Time')
plt.legend()
plt.grid(True)

plot_filename = f'attacks_detection_plot_{day_selection}.png'
# Guardar el plot como imagen
plt.savefig(plot_filename)

# Mostrar el plot
plt.show()


# %%

# %%
# Convertir a numpy arrays las listas involucradas
timestamps = np.array(metrics_df['timestamp'].tolist())
accuracy = np.array(metrics_df['accuracy'].tolist())
f1_score = np.array(metrics_df['f1'].tolist())
recall = np.array(metrics_df['recall'].tolist())
precision = np.array(metrics_df['precision'].tolist())
attacks_sum = np.array(metrics_df['true_negatives'] + metrics_df['false_negatives'])

# Tiempos en los que se reentrenó el modelo
retrain_timestamps = np.array(metrics_df[metrics_df['retrained'] == True]['timestamp'].tolist())

# Crear la figura y los subplots
plt.figure(figsize=(20, 15))

# Función para sombrear el área de ataques
def shade_attack_zones(ax, timestamps, attacks_sum):
    for i in range(len(timestamps) - 1):
        if attacks_sum[i] > 100:
            ax.axvspan(timestamps[i], timestamps[i+1], color='gray', alpha=0.3)

# Plot 1: Accuracy
plt.subplot(4, 1, 1)
plt.plot(timestamps, accuracy, label='Accuracy', color='purple', marker='o')
for retrain_time in retrain_timestamps:
    plt.axvline(x=retrain_time, color='blue', linestyle='--', label='Model Retrained')
shade_attack_zones(plt.gca(), timestamps, attacks_sum)
plt.xlabel('Time')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Over Time')
plt.legend()
plt.grid(True)

# Plot 2: F1-Score
plt.subplot(4, 1, 2)
plt.plot(timestamps, f1_score, label='F1-Score', color='orange', marker='o')
for retrain_time in retrain_timestamps:
    plt.axvline(x=retrain_time, color='blue', linestyle='--', label='Model Retrained')
shade_attack_zones(plt.gca(), timestamps, attacks_sum)
plt.xlabel('Time')
plt.ylabel('F1-Score')
plt.title('F1-Score Over Time')
plt.legend()
plt.grid(True)

# Plot 3: Recall
plt.subplot(4, 1, 3)
plt.plot(timestamps, recall, label='Recall', color='green', marker='o')
for retrain_time in retrain_timestamps:
    plt.axvline(x=retrain_time, color='blue', linestyle='--', label='Model Retrained')
shade_attack_zones(plt.gca(), timestamps, attacks_sum)
plt.xlabel('Time')
plt.ylabel('Recall')
plt.title('Recall Over Time')
plt.legend()
plt.grid(True)

# Plot 4: Precision
plt.subplot(4, 1, 4)
plt.plot(timestamps, precision, label='Precision', color='red', marker='o')
for retrain_time in retrain_timestamps:
    plt.axvline(x=retrain_time, color='blue', linestyle='--', label='Model Retrained')
shade_attack_zones(plt.gca(), timestamps, attacks_sum)
plt.xlabel('Time')
plt.ylabel('Precision')
plt.title('Precision Over Time')
plt.legend()
plt.grid(True)

# Ajustar el espacio entre los subplots
plt.tight_layout()

# Guardar el plot como imagen
plot_filename = f'model_metrics_with_attack_zones_{day_selection}.png'
plt.savefig(plot_filename)

# Mostrar el plot
plt.show()




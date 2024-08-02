import h2o
from h2o.automl import H2OAutoML
import pandas as pd

h2o.init()

# Load the data
df1 = pd.read_csv("CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv")
df1.columns = df1.columns.str.strip()
dataset = h2o.H2OFrame(df1, header=1)

print(dataset.col_names)
#concat tuesday
df2 = pd.read_csv("CICIDS2017/Tuesday-WorkingHours.pcap_ISCX.csv")
df2.columns = df2.columns.str.strip()
dataset2 = h2o.H2OFrame(df2, header=1)


#axis 0 is for concatenating rows, default to 1 for columns

train = dataset.concat(dataset2, axis=0)

#predictors and response
predictors = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min"
]



response = "Label"

# Split the data
df3 = pd.read_csv("CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv", low_memory=False)
df3.columns = df3.columns.str.strip()
valid = h2o.H2OFrame(df3, header=1)

#start AutoML only for DL and DRF models
aml = H2OAutoML(max_runtime_secs=60,
                seed=1234,  
                verbosity="info")

aml.train(x=predictors, y=response, training_frame=train, validation_frame=valid)

saving_model = None
saving_model = aml.leader

print(aml.leaderboard)
print(saving_model)
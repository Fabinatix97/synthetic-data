# 0. Laden der Bibliotheken

from sdv.datasets.demo import download_demo
from sdv.single_table import CTGANSynthesizer
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality, get_column_plot, get_column_pair_plot
import time

# 1. Laden der Daten

real_data, metadata = download_demo(
    modality='single_table',
    dataset_name='census_extended'
)
len(real_data)
real_data.head()

### Übersicht der Metadaten des ausgewählten Datensatzes
metadata.visualize()

# 2. Erstellen des Synthesizers

### Erstellung des Synthesizers unter Verwendung des CTGAN Algorithmus
synthesizer = CTGANSynthesizer(metadata)

### Trainieren des Synthesizers anhand der echten Daten (`real_data`)
start_time = time.time()
synthesizer.fit(
    data=real_data
)
end_time = time.time()
execution_time = end_time - start_time
minutes = int(execution_time // 60)
seconds = execution_time % 60

print(f"Trainingszeit: {minutes} Minuten und {seconds:.2f} Sekunden")

### Berechnete Verlustwerte für jede Epoche sowohl für den Generator als auch für den Diskriminator
synthesizer.get_loss_values()

### Optionale Visualisierung der Verlustwerte
fig = synthesizer.get_loss_values_plot()
fig.show()

# 3. Generieren synthetischer Daten

synthetic_data = synthesizer.sample(
    num_rows=5000
)

### Vergleich der realen Daten mit den synthetischen Daten
real_data[5:10]
synthetic_data.head()

# 4. Evaluation realer vs. synthetischer Daten
## 4.1 Diagnose

diagnostic = run_diagnostic(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata=metadata
)

## 4.2 Datenqualität

quality_report = evaluate_quality(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata=metadata
)

### Überblick über detaillierte Werte
quality_report.get_details('Column Shapes')

## 4.3 Anonymisierung

### Der Originaldatensatz enthielt einige sensible Spalten wie die beispielsweise die Wohnadresse oder der Geburtsort. In den synthetischen Daten sind diese Spalten vollständig anonymisiert - sie enthalten vollständig gefälschte Werte, die dem Format der Originaldaten entsprechen.
sensitive_column_names = ['dob', 'pob', 'address']
real_data[sensitive_column_names].head()
synthetic_data[sensitive_column_names].head()

## 4.4 Visualisierung der Daten

### Altersverteilung von realen und synthetischen Daten
fig = get_column_plot(
    real_data=real_data,
    synthetic_data=synthetic_data,
    column_name='age',
    metadata=metadata
)
fig.show()

### Verteilung der Ausbildungsjahre von realen und synthetischen Daten
fig = get_column_plot(
    real_data=real_data,
    synthetic_data=synthetic_data,
    column_name='education-num',
    metadata=metadata
)
fig.show()

### Verteilung des Ehestands von realen und synthetischen Daten
fig = get_column_plot(
    real_data=real_data,
    synthetic_data=synthetic_data,
    column_name='marital-status',
    metadata=metadata
)
fig.show()

### Korrelation zwischen Art der Beschäftigung und Ausbildungsdauer
fig = get_column_pair_plot(
    real_data=real_data,
    synthetic_data=synthetic_data,
    column_names=['education-num', 'occupation'],
    metadata=metadata
)
fig.show()

### Korrelation zwischen Ethnie und Ausbildungsdauer
fig = get_column_pair_plot(
    real_data=real_data,
    synthetic_data=synthetic_data,
    column_names=['education-num', 'race'],
    metadata=metadata
)
fig.show()

### Korrelation zwischen Einkommen und Ausbildungsdauer
fig = get_column_pair_plot(
    real_data=real_data,
    synthetic_data=synthetic_data,
    column_names=['education-num', 'income'],
    metadata=metadata
)
fig.show()

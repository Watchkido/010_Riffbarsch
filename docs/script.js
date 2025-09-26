// Smooth Scrolling für Navigationslinks
document.querySelectorAll('.nav a').forEach(link => {
    link.addEventListener('click', function(e) {
        e.preventDefault();
        
        // Alle Navigationslinks zurücksetzen
        document.querySelectorAll('.nav a').forEach(navLink => {
            navLink.classList.remove('active');
        });
        
        // Aktuellen Link als aktiv markieren
        this.classList.add('active');
        
        const targetId = this.getAttribute('href').substring(1);
        
        // Alle Sektionen zurücksetzen
        document.querySelectorAll('.vis-container').forEach(container => {
            container.classList.remove('active');
        });
        document.querySelectorAll('.text-content').forEach(content => {
            content.classList.remove('active');
        });
        
        // Zielabschnitt anzeigen
        document.getElementById(`${targetId}-vis`).classList.add('active');
        document.getElementById(`${targetId}-text`).classList.add('active');
        
        // Charts aktualisieren
        updateCharts(targetId);
    });
});

// Chart-Funktionen
function updateCharts(section) {
    switch(section) {
        case 'einleitung':
            createIntroChart();
            break;
        case 'pipeline':
            createPipelineChart();
            break;
        case 'modelle':
            createModelComparisonChart();
            break;
        case 'ergebnisse':
            createResultsChart();
            createConfusionMatrix();
            break;
    }
}

function createIntroChart() {
    const data = [{
        values: [35, 25, 20, 15, 5],
        labels: ['Thunfisch', 'Lachs', 'Barsch', 'Forelle', 'Andere'],
        type: 'pie',
        hole: 0.4,
        marker: {
            colors: ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        }
    }];
    
    const layout = {
        title: 'Verteilung der Fischarten im Datensatz',
        height: 300,
        showlegend: true
    };
    
    Plotly.newPlot('intro-chart', data, layout, {displayModeBar: false});
}

function createPipelineChart() {
    const data = [{
        x: ['Daten sammeln', 'Vorverarbeitung', 'Augmentation', 'Training', 'Evaluation'],
        y: [100, 85, 70, 60, 95],
        type: 'scatter',
        mode: 'lines+markers',
        line: {
            color: '#2ecc71',
            width: 3
        },
        marker: {
            size: 8,
            color: '#2ecc71'
        }
    }];
    
    const layout = {
        title: 'Daten-Pipeline - Qualität in jedem Schritt',
        xaxis: {title: 'Pipeline-Schritte'},
        yaxis: {title: 'Qualität (%)', range: [0, 100]},
        height: 300
    };
    
    Plotly.newPlot('pipeline-chart', data, layout, {displayModeBar: false});
}

function createModelComparisonChart() {
    const data = [
        {
            x: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            y: [92, 89, 91, 90],
            name: 'ResNet-18',
            type: 'bar',
            marker: {color: '#3498db'}
        },
        {
            x: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            y: [88, 85, 87, 86],
            name: 'YOLOv8n',
            type: 'bar',
            marker: {color: '#2ecc71'}
        },
        {
            x: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            y: [95, 93, 94, 93.5],
            name: 'Mask R-CNN',
            type: 'bar',
            marker: {color: '#e74c3c'}
        }
    ];
    
    const layout = {
        title: 'Modell-Vergleich nach Metriken',
        barmode: 'group',
        height: 300,
        yaxis: {
            title: 'Wert (%)',
            range: [0, 100]
        }
    };
    
    Plotly.newPlot('model-comparison-chart', data, layout, {displayModeBar: false});
}

function createResultsChart() {
    const data = [{
        x: ['ResNet-18', 'YOLOv8n', 'Mask R-CNN'],
        y: [92, 88, 95],
        type: 'bar',
        marker: {
            color: ['#3498db', '#2ecc71', '#e74c3c']
        }
    }];
    
    const layout = {
        title: 'Accuracy-Vergleich der Modelle',
        yaxis: {
            title: 'Accuracy (%)',
            range: [0, 100]
        },
        height: 300
    };
    
    Plotly.newPlot('results-chart', data, layout, {displayModeBar: false});
}

function createConfusionMatrix() {
    const z = [
        [45, 3, 2],
        [2, 48, 0],
        [1, 1, 43]
    ];
    
    const data = [{
        z: z,
        x: ['Thunfisch', 'Lachs', 'Barsch'],
        y: ['Thunfisch', 'Lachs', 'Barsch'],
        type: 'heatmap',
        colorscale: 'Blues',
        showscale: true
    }];
    
    const layout = {
        title: 'Confusion Matrix - Mask R-CNN',
        height: 300
    };
    
    Plotly.newPlot('confusion-matrix', data, layout, {displayModeBar: false});
}

// Bildanalyse-Funktion (Demo)
function analyzeImage() {
    const fileInput = document.getElementById('imageUpload');
    if (fileInput.files.length === 0) {
        alert('Bitte wählen Sie zuerst ein Bild aus');
        return;
    }
    
    // Hier würde die echte Analyse-Logik stehen
    alert('Bildanalyse gestartet! In einer echten Implementierung würde hier die Kommunikation mit Ihren ML-Modellen erfolgen.\n\nSimulierte Ergebnisse:\n- ResNet-18: Thunfisch (92% Sicherheit)\n- YOLOv8n: 3 Fische erkannt\n- Mask R-CNN: Pixelgenaue Segmentierung abgeschlossen');
}

// Initiale Charts laden
document.addEventListener('DOMContentLoaded', function() {
    createIntroChart();
});
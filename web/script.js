// Einfache Demo-Funktionalität
function analyzeImage() {
    const fileInput = document.getElementById('imageUpload');
    if (fileInput.files.length === 0) {
        alert('Bitte wählen Sie ein Bild aus');
        return;
    }
    
    // Hier würde die echte Analyse-Logik stehen
    alert('Bildanalyse gestartet! In einer echten Implementierung würde hier die Kommunikation mit Ihren ML-Modellen erfolgen.');
}

// Smooth Scroll für Navigation
document.querySelectorAll('nav a').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const targetId = this.getAttribute('href');
        document.querySelector(targetId).scrollIntoView({
            behavior: 'smooth'
        });
    });
});
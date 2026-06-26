document.addEventListener("DOMContentLoaded", () => {
    const currentWordEl = document.getElementById('current-word');
    const confidenceEl = document.getElementById('confidence');
    const sentenceEl = document.getElementById('sentence');
    const clearBtn = document.getElementById('clear-btn');
    const ttsToggle = document.getElementById('tts-toggle');
    
    // Web Speech API for Text-to-Speech
    const synth = window.speechSynthesis;
    let lastSpokenSentence = "";

    function speak(text) {
        if (!ttsToggle.checked) return;
        if (synth.speaking) synth.cancel(); // Interrupt to speak the latest
        
        const utterThis = new SpeechSynthesisUtterance(text);
        utterThis.rate = 0.9; // Slightly slower for clarity
        synth.speak(utterThis);
    }

    function fetchState() {
        fetch('/get_state')
            .then(response => response.json())
            .then(data => {
                // Update Current Live Prediction
                if (data.current_word) {
                    currentWordEl.textContent = data.current_word;
                    confidenceEl.textContent = Math.round(data.confidence * 100) + '%';
                    confidenceEl.style.opacity = 1;
                } else {
                    currentWordEl.textContent = "Waiting...";
                    confidenceEl.style.opacity = 0;
                }

                // Update Sentence
                if (data.sentence) {
                    sentenceEl.textContent = data.sentence;
                    
                    // Trigger TTS if a new word was just added
                    if (data.trigger_speak) {
                        speak(data.sentence);
                    }
                } else {
                    sentenceEl.textContent = "No signs detected yet.";
                }
            })
            .catch(error => console.error('Error fetching state:', error));
    }

    // Clear Button Handler
    clearBtn.addEventListener('click', () => {
        fetch('/clear')
            .then(() => {
                sentenceEl.textContent = "No signs detected yet.";
                if (synth.speaking) synth.cancel();
            });
    });

    // Poll the server state every 300ms
    setInterval(fetchState, 300);
});

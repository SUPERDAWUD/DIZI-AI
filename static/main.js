// Chat session management and model/personality switcher logic
let chatSessions = JSON.parse(localStorage.getItem('chatSessions') || '[]');
let currentSession = null;

function saveSession(session) {
    if (!session) return;
    chatSessions = chatSessions.filter(s => s.id !== session.id);
    chatSessions.unshift(session);
    localStorage.setItem('chatSessions', JSON.stringify(chatSessions));
    renderChatList();
}

function renderChatList() {
    const list = document.getElementById('chat-list');
    list.innerHTML = '';
    chatSessions.forEach(s => {
        const li = document.createElement('li');
        li.textContent = s.title || 'Chat ' + s.id;
        li.style.cursor = 'pointer';
        li.onclick = () => loadSession(s.id);
        list.appendChild(li);
    });
}

function loadSession(id) {
    const session = chatSessions.find(s => s.id === id);
    if (!session) return;
    currentSession = session;
    document.getElementById('chat-box').innerHTML = session.messages.map(m => `
${m.user}: ${m.text}
`).join('');
}

function newSession() {
    const id = Date.now();
    currentSession = { id, title: '', messages: [] };
    saveSession(currentSession);
}

// Call newSession on load if none
if (!chatSessions.length) newSession();
else {
    currentSession = chatSessions[0];
    renderChatList();
}

// --- Model/Personality Switcher Logic ---
function switchModel() {
    const model = document.getElementById('model-select').value;
    const personality = document.getElementById('personality-select').value;
    localStorage.setItem('selectedModel', model);
    localStorage.setItem('selectedPersonality', personality);
    alert('Switched to ' + model + ' model with ' + personality + ' personality!');
}

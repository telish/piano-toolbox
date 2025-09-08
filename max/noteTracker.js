// Max JS: noteTracker.js
// Receives lists: pitch velocity
// Tracks currently playing notes
// If a note-on arrives for a pitch already playing, sends note-off first

// Dictionary to track currently playing notes
var activeNotes = {}; // activeNotes[pitch] = velocity

// Function called when a list comes in
function list() {
    var args = arrayfromargs(arguments);
    var pitch = args[0];
    var velocity = args[1];

    if (velocity > 0) { // note-on
        if (activeNotes[pitch]) {
            // Pitch already playing, send note-off first
            outlet(0, [pitch, 0]);
        }
        // Send the new note-on
        outlet(0, [pitch, velocity]);
        activeNotes[pitch] = velocity;
    } else { // note-off
        // Only send if the pitch is actually active
        if (activeNotes[pitch]) {
            outlet(0, [pitch, 0]);
            delete activeNotes[pitch];
        }
    }
}

function clear() {
    for (var pitch in activeNotes) {
        outlet(0, [parseInt(pitch), 0]);
    }
    activeNotes = {};
}
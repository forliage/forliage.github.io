document.addEventListener('DOMContentLoaded', function() {
    // Get the modal element
    const modal = document.getElementById('about-me-modal');

    // Get the button that opens the modal
    const btn = document.getElementById('about-me-btn');

    // Get the <span> element that closes the modal
    const span = document.getElementsByClassName('close-button')[0];

    // When the user clicks the button, open the modal
    if (btn && modal) {
        btn.onclick = function(event) {
            event.preventDefault(); // Prevent default link behavior if it's an <a> tag
            modal.style.display = 'block';
        }
    }

    // When the user clicks on <span> (x), close the modal
    if (span && modal) {
        span.onclick = function() {
            modal.style.display = 'none';
        }
    }

    // When the user clicks anywhere outside of the modal content, close it
    window.onclick = function(event) {
        if (event.target == modal) {
            modal.style.display = 'none';
        }
    }
});

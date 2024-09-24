/*document.getElementById('loan-form').addEventListener('submit', function(event) {
    event.preventDefault();
  
    const formData = new FormData(this);
  
    fetch('/predict', {
      method: 'POST',
      body: formData
    })
    .then(response => response.json())
    .then(data => {
      const popupMessage = document.getElementById('popup-message');
      if (data.error) {
        popupMessage.classList.add('error');
        popupMessage.innerHTML = `Error: ${data.error}`;
      } else {
        popupMessage.classList.add('success');
        popupMessage.innerHTML = `LOAN STATUS: ${data.result}`;
      }
      popupMessage.style.display = 'block'; // Show the popup message
    })
    .catch(error => {
      const popupMessage = document.getElementById('popup-message');
      popupMessage.classList.add('error');
      popupMessage.innerHTML = `An error occurred: ${error}`;
      popupMessage.style.display = 'block'; // Show the popup message
    });
  });*/
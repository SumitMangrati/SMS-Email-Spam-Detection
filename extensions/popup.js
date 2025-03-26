document.getElementById("checkSpam").addEventListener("click", function () {
    let message = document.getElementById("message").value;
    let resultBox = document.getElementById("result");

    if (!message) {
        resultBox.innerText = "Please enter a message.";
        resultBox.style.display = "block";
        resultBox.className = "";
        return;
    }

    fetch("https://sms-spam-detector-zhys.onrender.com/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {
        resultBox.innerText = "Result: " + data.result;
        resultBox.style.display = "block";
        
        if (data.result === "Spam") {
            resultBox.className = "spam";
        } else {
            resultBox.className = "not-spam";
        }
    })
    .catch(error => {
        console.error("Error:", error);
        resultBox.innerText = "Error connecting to the server.";
        resultBox.style.display = "block";
        resultBox.className = "";
    });
});

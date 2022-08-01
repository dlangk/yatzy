let d = new Date();


$.get("http://localhost:8080/", function (data, status) {
    alert("Data: " + data + "\nStatus: " + status);
});

document.body.innerHTML = "<h1>Today's date is " + d + "</h1>"
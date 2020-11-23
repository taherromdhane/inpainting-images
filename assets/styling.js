
function waitForElement(id, callback){
    var poops = setInterval(function() {
        if(document.getElementById(id)) {
            clearInterval(poops);
            callback();
        }
    }, 100);
}

waitForElement("canvas_div", function() {
    buttons = document.querySelectorAll("#canvas_div button");
    buttons.forEach(element => {
        element.classList.add("btn");
        console.log(element);
        if (element.getAttribute("title") === "Save") {
            element.style.color = null;
            element.classList.add("btn-outline-primary");
        }
        else {
            element.classList.add("btn-outline-dark");
        }
    });
});
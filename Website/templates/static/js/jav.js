
var loadFile = function(event) {
	var image = document.getElementById('file');
	image.src = URL.createObjectURL(event.target.files[0]);
};

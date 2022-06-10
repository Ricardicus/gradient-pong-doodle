// handling the focus of the keylistener (solution found in thread on stackoverflow.com)
var lastDownTarget, canvas;
var radiuses = [
	[90, 70, 50, 30],
	[90, 70, 50, 30]
]; // Percent from 0 to R
var startPos = [
	[10, 10],
	[90, 10]
]; // percent of X and Y
var targetPos = [50, 50]; // Percent from 0-X and 0-Y

// Total pixel map size, max unit of distance, set by reading canvas properties
var XTot;
var YTot;
var R;

// Percent of 2 * PI
var omegas = [[], []];
var omegasStart = 35; // Initial value for all angles (percent of 2*PI)

// Optimization choices
var ADAM = Symbol("ADAM");
var BASIC = Symbol("BASIC");
var ADAGRAD = Symbol("ADAGRAD");

var optimizationStrategy = [BASIC, BASIC];
var optimizationStop = [4, 4];
var optimizationHit = [0, 0];

// Gradient decent delta update
var learningRate = [0.003288, 0.003288];

// Some thresholds for input
var learningRateMin = 0.00000001;
var learningRateMax = 100.0;

var updateSpeed = 5;

// Adam optimization algorithm variables
var adam_beta_1 = [0.27, 0.27];
var adam_beta_1_min = 0.0001;
var adam_beta_1_max = 0.98;
var adam_beta_2 = [0.99, 0.99];
var adam_beta_2_min = 0.001;
var adam_beta_2_max = 0.9999;
var adam_epsilon = 0.000001;
var adam_v = [[], []]; // one per omega
var adam_momentum_omegas = [[], []]; // one per omega

// Ada grad optimization parameters
var adagrad_G = [[], []];

// Regularization
var lambda = 0.0;

// To avoid div by zero
var epsilon = 0.00000000001;

// Target position movement
var dx = 0.03;
var dxStart = 0.05;
var dy = 0;
var targetAcceleration = 1.6;
var targetMaxSpeed = 0.3;

// Score
var scoreboard = [0, 0];

function updateLearningRateBar(arm) {
	document.getElementById("arm-" + (arm + 1) + "-label-learning-rate").innerHTML = "Learning rate: " +
		Number.parseFloat(learningRate[arm]).toPrecision(4);

	var targetE = Math.log10(learningRate[arm]);
	var minv = Math.log10(learningRateMin);
	var maxv = Math.log10(learningRateMax);
	var scale = (maxv - minv) / 100.0;

	var val = parseInt((targetE - minv) / scale);
	document.getElementById("arm-" + (arm + 1) + "-learning-rate").value = val;
}

function newOptStrat(arm) {
	if (document.getElementById("arm-" + (arm + 1) + "-adam-select").checked) {
		optimizationStrategy[arm] = ADAM;
		learningRate[arm] = 0.3943;
		updateLearningRateBar(arm);

	} else if (document.getElementById("arm-" + (arm + 1) + "-adagrad-select").checked) {
		learningRate[arm] = 0.3943;
		updateLearningRateBar(arm);
		optimizationStrategy[arm] = ADAGRAD;
	} else {
		optimizationStrategy[arm] = BASIC;
	}
}

function newADAMBeta1(arm) {
	var minv = Math.log10(adam_beta_1_min);
	var maxv = Math.log10(adam_beta_1_max);
	var scale = (maxv - minv) / 100.0;

	adam_beta_1[arm] = Math.pow(10, minv + scale * (
		document.getElementById("arm-" + (arm + 1) + "-adam-beta-1").value));
	document.getElementById("arm-" + (arm + 1) + "-label-adam-beta-1").innerHTML = "ADAM β1: " +
		Number.parseFloat(adam_beta_1[arm]).toPrecision(4);
}

function newADAMBeta2(arm) {
	var minv = Math.log10(adam_beta_2_min);
	var maxv = Math.log10(adam_beta_2_max);
	var scale = (maxv - minv) / 100.0;

	adam_beta_2[arm] = Math.pow(10, minv + scale * (
		document.getElementById("arm-" + (arm + 1) + "-adam-beta-2").value));
	document.getElementById("arm-" + (arm + 1) + "-label-adam-beta-2").innerHTML = "ADAM β2: " +
		Number.parseFloat(adam_beta_2[arm]).toPrecision(4);
}

function newLearningRate(arm) {
	var minv = Math.log10(learningRateMin);
	var maxv = Math.log10(learningRateMax);
	var scale = (maxv - minv) / 100.0;

	learningRate[arm] = Math.pow(10, minv + scale * (
		document.getElementById("arm-" + (arm + 1) + "-learning-rate").value));
	document.getElementById("arm-" + (arm + 1) + "-label-learning-rate").innerHTML = "Learning rate: " +
		Number.parseFloat(learningRate[arm]).toPrecision(4);
}

function newOmegas(arm) {
	for (var i = 0; i < omegas[0].length; i++) {
		omegas[arm][i] = document.getElementById("arm-" + (arm + 1) + "-omega-" + (i + 1)).value;
	}
}

function resetOptimization() {
	adam_v = [[], []];
	adam_momentum_omegas = [[], []];
	adagrad_G = [[], []];
	for (var i = 0; i < 2; i++) {
		omegas[i].forEach((a) => {
			adam_v[i].push(0.0);
			adam_momentum_omegas[i].push(0.0);
			adagrad_G[i].push(0.0);
		});
	}
}


function resetOmegas(arm) {
	for (var i = 0; i < omegas[arm].length; i++) {
		omegas[arm][i] = omegasStart;
	}
}

function updateTargetPos(canvas, evt) {
	var rect = canvas.getBoundingClientRect();
	var ctx = canvas.getContext("2d");
	XTot = canvas.width;
	YTot = canvas.height;

	var mousePos = {
		x: evt.clientX - rect.left,
		y: evt.clientY - rect.top
	};

	if (mousePos.x > 0 && mousePos.x < XTot &&
		mousePos.y > 0 && mousePos.y < YTot) {
		targetPos = [100 * mousePos.x / XTot, 100 * mousePos.y / YTot];
		resetOptimization();
	}
}

function norm(p1, p2) {
	var sum = 0.0;
	for (var i = 0; i < p1.length; i++) {
		sum += Math.abs(p1[i] - p2[i]);
	}
	return sum;
}

function omega_sum_order(n, omegas) {
	var sum = -(n - 1) * Math.PI;
	for (var i = 0; i < n; i++) {
		sum += omegas[i] * 0.01 * 2 * Math.PI;
	}
	return sum;
}

function de_dpnx_domega_q(omegas, radiuses, R, q, n) {
	if (q > n) return 0.0;
	return de_dpnx_domega_q(omegas, radiuses, R, q, n - 1) - radiuses[n - 1] * 0.01 * R * Math.sin(omega_sum_order(n, omegas)) * Math.PI * 2 * 0.01;
}

function de_dpny_domega_q(omegas, radiuses, R, q, n) {
	if (q > n) return 0.0;
	return de_dpny_domega_q(omegas, radiuses, R, q, n - 1) + radiuses[n - 1] * 0.01 * R * Math.cos(omega_sum_order(n, omegas)) * Math.PI * 2 * 0.01;
}

function de_domega_q(point, target, radiuses, omegas, R, q) {
	var n = omegas.length;
	return (point[0] - target[0]) * de_dpnx_domega_q(omegas, radiuses, R, q, n) +
		(point[1] - target[1]) * de_dpny_domega_q(omegas, radiuses, R, q, n);
}

function p_n_x(omegas, radiuses, R, n) {
	var pnx = radiuses[n - 1] * 0.01 * R * Math.cos(omega_sum_order(n, omegas));
	if (n > 1) {
		pnx += p_n_x(omegas, radiuses, R, n - 1);
	}
	return pnx;
}

function p_n_y(omegas, radiuses, R, n) {
	var pny = radiuses[n - 1] * 0.01 * R * Math.sin(omega_sum_order(n, omegas));
	if (n > 1) {
		pny += p_n_y(omegas, radiuses, R, n - 1);
	}
	return pny;
}

function mean_square_loss(pos, target) {
	return Math.pow(norm(pos, target), 2) * 0.5;
}

function populateValues() {

	for (var arm = 0; arm < 2; arm++) {
		var newHTML = "<pre>Angles Ω:s</pre>";
		for (var i = 0; i < omegas[0].length; i++) {
			newHTML += '<input type="range" oninput="newOmegas(' + arm + ');"\
         min="0" max="100" id="arm-'+ (arm + 1) + '-omega-' + (i + 1) + '">\
  <label id="arm-'+(arm+1)+'-label-bar-'+ (i + 1) + '" for="arm-' + (arm + 1) + '-omega-" ' + (i + 1) + '>Ω' + (i + 1) + '</label><br/>';
		}
		document.getElementById("arm-" + (arm + 1) + "-omega-bars").innerHTML = newHTML;
		document.getElementById("arm-" + (arm + 1) + "-label-learning-rate").innerHTML = "Learning rate: " + Number.parseFloat(learningRate[arm]).toPrecision(4);
		document.getElementById("arm-" + (arm + 1) + "-label-adam-beta-1").innerHTML = "ADAM β1: " + Number.parseFloat(adam_beta_1[arm]).toPrecision(4);
		document.getElementById("arm-" + (arm + 1) + "-label-adam-beta-2").innerHTML = "ADAM β2: " + Number.parseFloat(adam_beta_2[arm]).toPrecision(4);

		var targetE = Math.log10(learningRate[arm]);
		var minv = Math.log10(learningRateMin);
		var maxv = Math.log10(learningRateMax);
		var scale = (maxv - minv) / 100.0;

		var val = parseInt((targetE - minv) / scale);
		document.getElementById("arm-" + (arm + 1) + "-learning-rate").value = val;

		targetE = Math.log10(adam_beta_1[arm]);
		var minv = Math.log10(adam_beta_1_min);
		var maxv = Math.log10(adam_beta_1_max);
		var scale = (maxv - minv) / 100.0;

		val = parseInt((targetE - minv) / scale);
		document.getElementById("arm-" + (arm + 1) + "-adam-beta-1").value = val;

		targetE = Math.log10(adam_beta_2[arm]);
		minv = Math.log10(adam_beta_2_min);
		maxv = Math.log10(adam_beta_2_max);
		scale = (maxv - minv) / 100.0;

		val = parseInt((targetE - minv) / scale);
		document.getElementById("arm-" + (arm + 1) + "-adam-beta-2").value = val;

		// check optimization
		if (document.getElementById("arm-" + (arm + 1) + "-adam-select").checked) {
			optimizationStrategy[arm] = ADAM;
		} else if (document.getElementById("arm-" + (arm + 1) + "-adagrad-select").checked) {
			optimizationStrategy[arm] = ADAGRAD;
		} else {
			optimizationStrategy[arm] = BASIC;
		}
	}
}

window.onload = function(e) {
	canvas = document.getElementById("tcanvas");

	document.addEventListener('mousedown', function(event) {
		lastDownTarget = event.target;
		updateTargetPos(canvas, event);
	}, false);

	document.addEventListener('keydown', function(event) {
		if (lastDownTarget == canvas) {
			keypressed(event);
		}
	}, false);

	// By default, omegas go to 35
	for (var i = 0; i < radiuses[0].length; i++) {
		for (var n = 0; n < omegas.length; n++) {
			omegas[n].push(omegasStart);
		}
	}

	populateValues();
	resetOptimization();

	var ctx = canvas.getContext("2d");
	XTot = canvas.width;
	YTot = canvas.height;
	R = YTot * 0.5;

	setInterval(loop, updateSpeed);
}

function getPoints(arm) {
	var points = [];
	for (var n = 1; n <= omegas[arm].length; n++) {
		points.push(
			[
				parseInt(
					startPos[arm][0] * 0.01 * XTot + p_n_x(omegas[arm], radiuses[arm], R, n)
				),
				parseInt(
					startPos[arm][1] * 0.01 * YTot + p_n_y(omegas[arm], radiuses[arm], R, n)
				)
			]
		);
	}
	return points;
}

function updateOmegas(arm, points) {
	// The objective is for x3 to reach the target
	var point = points[omegas[arm].length - 1];
	var target = [targetPos[0] * 0.01 * XTot, targetPos[1] * 0.01 * YTot];
	var error = mean_square_loss(point, target);

	for (var i = 0; i < omegas[arm].length; i++) {
		document.getElementById("arm-" + (arm + 1) + "-omega-" + (i + 1)).value = omegas[arm][i];
	}
	if (arm == 0 && dx > 0) {
		// The ball is moving from us, it is OK
		relaxArm(arm);
		return;
	}
	if (arm == 1 && dx < 0) {
		// The ball is moving from us, it is OK
		relaxArm(arm);
		return;
	}

	if ( error <= optimizationStop[arm]) {
		dx = (-1) * dx * targetAcceleration; // going faster and faster
		if ( Math.abs(dx) > targetMaxSpeed ) {
			dx = targetMaxSpeed * Math.sign(dx);
		}
		dy = (Math.random() - 0.5) * 2.0 * 0.5
		// reset optimization
		resetOptimization();
	}

	document.getElementById("arm-" + (arm + 1) + "-loss").innerHTML = Number.parseFloat(error).toPrecision(4);
	var de_domegas = [];
	for (var i = 0; i < omegas[arm].length; i++) {
		var de_domega_i = de_domega_q(point, target, radiuses[arm], omegas[arm], R, i + 1);
		de_domegas.push(de_domega_i);
	}

	if (optimizationStrategy[arm] == BASIC) {
		for (var i = 0; i < omegas[arm].length; i++) {
			omegas[arm][i] = omegas[arm][i] - learningRate[arm] * de_domegas[i] - lambda * Math.abs(omegas[arm][i]);
		}
	} else if (optimizationStrategy[arm] == ADAM) {
		for (var i = 0; i < omegas[arm].length; i++) {
			adam_v[arm][i] = adam_beta_2[arm] * adam_v[arm][i] + (1.0 - adam_beta_2[arm]) * Math.pow(de_domegas[i], 2);
			adam_momentum_omegas[arm][i] = adam_beta_1[arm] * adam_momentum_omegas[arm][i] + (1.0 - adam_beta_1[arm]) * de_domegas[i];
			vhat = adam_v[arm][i] / (1.0 - adam_beta_2[arm]);
			mhat = adam_momentum_omegas[arm][i] / (1.0 - adam_beta_1[arm]);
			omegas[arm][i] = omegas[arm][i] - learningRate[arm] * mhat / (Math.sqrt(vhat) + adam_epsilon);
		}
	} else if (optimizationStrategy[arm] == ADAGRAD) {
		for (var i = 0; i < omegas[arm].length; i++) {
			var g = de_domegas[i];
			adagrad_G[arm][i] += Math.pow(g, 2);
			omegas[arm][i] = omegas[arm][i] - learningRate[arm] * ((1.0 / Math.sqrt(adagrad_G[arm][i] + epsilon)) * de_domegas[i]);
		}
	} else {
		console.log("no opt..");
	}
}

function debugPrint(arm) {
	for (var i = 0; i < omegas[arm].length; i++) {
		document.getElementById("arm-"+(arm+1)+"-label-bar-" + (i + 1)).innerHTML = "Ω" + (i + 1) + ": " + Number.parseFloat(omegas[arm][i] * 0.01 * 360).toFixed(2) + " °";
	}
}

var loopCount = 0;
function loop() {
	var ctx = canvas.getContext("2d");
	var points = [getPoints(0), getPoints(1)];
	ctx.clearRect(0, 0, canvas.width, canvas.height);
	for (var i = 0; i < points.length; i++) {
		drawArm(ctx, i, points[i]);
		updateOmegas(i, points[i]);
		debugPrint(i);
	}
	drawTarget(ctx);
	moveTargetPos(ctx);
	updateScoreBoard();
	loopCount = (loopCount + 1) % 100000;
}

function relaxArm(arm) {
	for (var i = 0; i < omegas[arm].length; i++) {
		omegas[arm][i] = omegas[arm][i] + (35 - omegas[arm][i]) * 0.01;
	}
}

function updateScoreBoard() {
	document.getElementById("scoreboard").innerHTML = "Score: " + scoreboard[0] + " - " + scoreboard[1];
}

// the timer identifier used for handling the restart calls.  
var timerId = 0;

// drawing the game on the canvas according to the content of the nxn-matrix 'blocks'
function drawArm(ctx, arm, points) {
	ctx.beginPath();
	ctx.moveTo(parseInt(startPos[arm][0] * XTot * 0.01), parseInt(startPos[arm][1] * YTot * 0.01));
	if (typeof points === "undefined") return;
	for (var i = 0; i < points.length; i++) {
		ctx.lineTo(points[i][0], points[i][1]);
	}
	ctx.stroke();
	ctx.beginPath();
	ctx.arc(points[points.length - 1][0], points[points.length - 1][1], 7, 0, 2 * Math.PI, false);
	ctx.fillStyle = 'black';
	ctx.fill();
	ctx.closePath();
}

function drawTarget(ctx) {
	// Draw target
	ctx.beginPath();
	ctx.arc(targetPos[0] * 0.01 * XTot, targetPos[1] * 0.01 * YTot, 4, 0, 2 * Math.PI, false);
	ctx.fillStyle = 'red';
	ctx.fill();
	ctx.stroke();
	ctx.closePath();
}

function moveTargetPos(ctx) {
	var newTX = (targetPos[0] + dx);
	var newTY = (targetPos[1] + dy);
	if (newTX > 100.0 || newTX < 0) {
		dx = -1 * dx;
	}
	if (newTY > 100 || newTY < 0) {
		dy = -1 * dy;
	}
	if (newTY > 100) newTY = 100;
	if (newTY < 0) newTY = 0;
	if (newTX > 100) {
		newTX = 100;
		scoreboard[0]++;
		dx = Math.sign(dx) * dxStart;
	        resetOptimization();
	}
	if (newTX < 0) {
		newTX = 0;
		scoreboard[1]++;
		dx = Math.sign(dx) * dxStart;
		resetOptimization();
	}
	targetPos[0] = newTX;
	targetPos[1] = newTY;
}


// key pressed handling
function keypressed(event) {
	var y = event.keyCode;
}



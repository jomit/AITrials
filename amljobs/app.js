var http = require("http");

function callApi(options, data, callback) {
    var req = http.request(options, function (res) {
        res.setEncoding('utf8');
        var responseData = '';

        res.on('data', function (result) {
            responseData = result;
        });

        res.on('end', function () {
            callback(JSON.parse(responseData));
        });
    });
    req.write(JSON.stringify(data));
    req.end();
}

function cleanupResponse(response){
    var dataValues = response.substring(2,response.length-2).split('.');
    var finalValues = []
    dataValues.forEach(element => {
        if(element !== "") {
            finalValues.push(parseInt(element.trim()));
        }
    });
    return finalValues;
}

var options = {
    host: "<IP>",
    port: 80,
    path: '/api/v1/service/jvnewservice/score',
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer <Token>'
    }
};

var inputData = {
    "inputData": [[20, 50000],[32,100000]]
}

callApi(options,inputData,function(response){
    var responseData = JSON.parse(response);
    console.log(responseData[0]);
    if(responseData[0] === 1){
         console.log("Do something...");
    }
});




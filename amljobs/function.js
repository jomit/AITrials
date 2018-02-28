var http = require("http");

module.exports = function (context, req) {
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
    callApi(options,req.body,function(response){
        var responseData = JSON.parse(response);
        context.log("Response => " + responseData);
        context.res = {
            body: responseData
        };
        context.done();
    });
};

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
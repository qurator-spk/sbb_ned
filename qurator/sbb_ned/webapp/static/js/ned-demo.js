$(document).ready(function(){

    $('#nerform').submit(
        function(e){
            e.preventDefault();

            update();
        }
    );

    let url_params = new URLSearchParams(window.location.search);

    let do_update=false;

    if (url_params.has('text')) {

        let text = decodeURIComponent(url_params.get('text'))

        $('#inputtext').val(text);

        do_update = true;

        window.history.replaceState({}, '', `${location.pathname}`);
    }

    //task_select()

    if (do_update) update();

});

function update() {

    //var task = $('#task').val();
    //var model_id = $('#model').val();
    let input_text = $('#inputtext').val()

    if (input_text.length < 30000) {

        var url_params = new URLSearchParams(window.location.search);

        url_params.set('text', encodeURIComponent(input_text))

        window.history.replaceState({}, '', `${location.pathname}?${url_params}`);
    }
    else {
        window.history.replaceState({}, '', `${location.pathname}`);
    }

    runNER(input_text,
        function (result) {
            showNERText(result);

            console.log(result);

            runNED(result,
                function(ned_result) {
                    console.log(ned_result);
                });
        }
    );

    //do_task(task, model_id, input_text);
}
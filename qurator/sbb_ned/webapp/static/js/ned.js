
function runNER (input_text, onSuccess) {

    let post_data = { "text" : input_text }

    $.ajax({
            url:  "../ner/ner/1" ,
            data: JSON.stringify(post_data),
            type: 'POST',
            contentType: "application/json",
            success: onSuccess,
            error: function(error) {
                console.log(error);
            }
        }
    );
}

function runNED (input, onSuccess) {

    let post_data = input;

    $.ajax({
            url:  "ned" ,
            data: JSON.stringify(post_data),
            type: 'POST',
            contentType: "application/json",
            success: onSuccess,
            error: function(error) {
                console.log(error);
            },
            timeout: 360000
        }
    );
}



function showNERText( data ) {

    var text_region_html =
        `<div class="card">
            <div class="card-header">
                Ergebnis:
            </div>
            <div class="card-block">
                <div id="ner-text" style="overflow-y:scroll;height: 55vh;"></div>
            </div>
        </div>`;

    text_html = ""
    data.forEach(
        function(sentence) {
            sentence.forEach(
                function(token) {

                     if (text_html != "") text_html += ' '

                     if (token.prediction == 'O')
                        text_html += token.word
                     else if (token.prediction.endsWith('PER'))
                        text_html += '<font color="red">' + token.word + '</font>'
                     else if (token.prediction.endsWith('LOC'))
                        text_html += '<font color="green">' + token.word + '</font>'
                     else if (token.prediction.endsWith('ORG'))
                        text_html += '<font color="blue">' + token.word + '</font>'
                })
             text_html += '<br/>'
        }
    )
    $("#result-text").html(text_region_html)
    $("#ner-text").html(text_html)
    //$("#legende").html(legende_html)
}

function do_task(task, model_id, input_text) {

    var post_data = { "text" : input_text }

    var text_region_html =
        `<div class="card">
            <div class="card-header">
                Ergebnis:
            </div>
            <div class="card-block">
                <div id="textregion" style="overflow-y:scroll;height: 55vh;"></div>
            </div>
        </div>`;

    var legende_html =
         `<div class="card">
            <div class="card-header">
                Legende:
                <div class="ml-2" >[<font color="red">Person</font>]</div>
                <div class="ml-2" >[<font color="green">Ort</font>]</div>
                <div class="ml-2" >[<font color="blue">Organisation</font>]</div>
                <div class="ml-2" >[keine Named Entity]</div>
            </div>
        </div>`;

    var spinner_html =
        `<div class="d-flex justify-content-center">
            <div class="spinner-border align-center" role="status">
                <span class="sr-only">Loading...</span>
            </div>
         </div>`;

    $("#legende").html("");

    if (task == "fulltext") {
        $("#resultregion").html(text_region_html)
        $("#textregion").html(input_text)
    }
    else if (task == "tokenize") {

        $("#resultregion").html(spinner_html)

        $.ajax(
            {
            url:  "tokenized",
            data: JSON.stringify(post_data),
            type: 'POST',
            contentType: "application/json",
            success:
                function( data ) {
                    text_html = ""
                    data.forEach(
                        function(sentence) {

                            text_html += JSON.stringify(sentence)

                            text_html += '<br/>'
                        }
                    )
                    $("#resultregion").html(text_region_html)
                    $("#textregion").html(text_html)
                    $("#legende").html(legende_html)
                }
            ,
            error:
                function(error) {
                    console.log(error);
                }
            })
    }
    else if (task == "ner") {

        $("#resultregion").html(spinner_html)

        $.ajax({
            url:  "ner/" + model_id,
            data: JSON.stringify(post_data),
            type: 'POST',
            contentType: "application/json",
            success:
                function( data ) {
                    text_html = ""
                    data.forEach(
                        function(sentence) {
                            sentence.forEach(
                                function(token) {

                                     if (text_html != "") text_html += ' '

                                     if (token.prediction == 'O')
                                        text_html += token.word
                                     else if (token.prediction.endsWith('PER'))
                                        text_html += '<font color="red">' + token.word + '</font>'
                                     else if (token.prediction.endsWith('LOC'))
                                        text_html += '<font color="green">' + token.word + '</font>'
                                     else if (token.prediction.endsWith('ORG'))
                                        text_html += '<font color="blue">' + token.word + '</font>'
                                })
                             text_html += '<br/>'
                        }
                    )
                    $("#resultregion").html(text_region_html)
                    $("#textregion").html(text_html)
                    $("#legende").html(legende_html)
                }
            ,
            error: function(error) {
                console.log(error);
            }
        });
     }
     else if (task == "bert-tokens") {
        $("#resultregion").html(spinner_html);

        $.ajax(
            {
            url:  "ner-bert-tokens/" + model_id,
            data: JSON.stringify(post_data),
            type: 'POST',
            contentType: "application/json",
            success:
                function( data ) {
                    text_html = ""
                    data.forEach(
                        function(sentence) {
                            sentence.forEach(
                                function(part) {

                                     if (text_html != "") text_html += ' '

                                     text_html += part.token + "(" + part.prediction + ")"
                                })
                             text_html += '<br/>'
                        }
                    )
                    $("#resultregion").html(text_region_html)
                    $("#textregion").html(text_html)
                    $("#legende").html(legende_html)
                }
            ,
            error:
                function(error) {
                    console.log(error);
                }
            })
     }
}
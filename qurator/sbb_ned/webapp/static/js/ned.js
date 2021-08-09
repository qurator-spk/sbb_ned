
function NED() {

    var that = null;

    var ner_result = null;
    var ner_parsed = null;
    var ned_result = { };

    var spinner_html =
            `<div class="d-flex justify-content-center">
                <div class="spinner-border align-center" role="status">
                    <span class="sr-only">Loading...</span>
                </div>
             </div>`;

    function runNER (ner_url, input_text, onSuccess) {

        $("#result-entities").html("");
        $("#result-text").html(spinner_html);

        let post_data = { "text" : input_text };

        $.ajax({
                url:  ner_url,
                data: JSON.stringify(post_data),
                type: 'POST',
                contentType: "application/json",
                success:
                    function(result) {
                        ner_result = result;
                        onSuccess(result);

                        console.log(result);
                    },
                error:
                    function(error) {
                        console.log(error);
                    }
            }
        );
    }

    function parseNER (parse_url, input, onSuccess) {

        let post_data = input;

        $.ajax({
                url:  parse_url,
                data: JSON.stringify(post_data),
                type: 'POST',
                contentType: "application/json",
                success:
                    function(result) {
                        ner_parsed = result;
                        onSuccess(result);
                    },
                error:
                    function(error) {
                        console.log(error);
                    },
                timeout: 360000
            }
        );
    }

    let ned_request_counter = 0;
    let ned_requested = { };

    function runNED (input, onSuccess) {

        if (ner_parsed == null) {
            console.log('Parsed NER data missing.');
            return;
        }

        let keys = Object.keys(input);

        if (keys in ned_requested) return;

        ned_requested[keys] = true;
        ned_request_counter++;

        let post_data = input;

        (function(current_counter) {
            $.ajax(
                {
                    url:  that.ned_url,
                    data: JSON.stringify(post_data),
                    type: 'POST',
                    contentType: "application/json",
                    success:
                        function(result) {
                            Object.assign(ned_result, result);

                            if (current_counter < ned_request_counter) return;

                            onSuccess(result);
                        },
                    error:
                        function(error) {
                            console.log(error);
                        },
                    timeout: 360000
                }
            );
        })(ned_request_counter);
    }

    function makeResultList(entities) {
        var entities_html = "";

        entities.forEach(
            function(candidate, index) {

                if (index > 10) return;

                //if (Number(candidate[1]) < 0.1) return;

                entities_html += '<a href="https://de.wikipedia.org/wiki/' + candidate[0] + '">'
                                        + candidate[0] + '</a> '
                                        + '(' + Number(candidate[1]['proba_1']).toFixed(2)
                                        + ', <a href="https://www.wikidata.org/wiki/' + candidate[1]['wikidata'] + '">'
                                        + candidate[1]['wikidata'] + '</a>'
                                        + ')' +' <br/>';
            }
        );

        $("#result-entities").html(entities_html);
    }

    function selectEntity(entity, onSuccess) {

        $("#result-entities").html(spinner_html);

        console.log(entity);

        if (entity in ned_result) {
            if ('ranking' in ned_result[entity]) {
                makeResultList(ned_result[entity]['ranking']);
                onSuccess();
                return;
            }
            else {
                $("#result-entities").html("NOT FOUND");
            }
        }

        if (!(entity in ner_parsed) ){
            console.log(entity)
            $("#result-entities").html("NO NER DATA.");
            return;
        }

        var input = {};
        input[entity] = ner_parsed[entity];

        runNED(input,
            function() {
                if (entity in ned_result) {
                    if ('ranking' in ned_result[entity]) {
                        makeResultList(ned_result[entity]['ranking']);

                        onSuccess();
                    }
                    else {
                        $("#result-entities").html("NOT FOUND");
                    }
                }
                else {
                    $("#result-entities").html("NOT FOUND");
                }
            }
        );

        // console.log(ned_result[entity]);
    }

    function showNERText( data ) {

        function getColor(entity_type) {
            if (entity_type.endsWith('PER'))
                return "red"
            else if (entity_type.endsWith('LOC'))
                return "green"
            else if (entity_type.endsWith('ORG'))
                return "blue"
        }

        var text_region_html =
            `<div class="card">
                <div class="card-header">
                    Ergebnis:
                </div>
                <div class="card-block">
                    <div id="ner-text" style="overflow-y:scroll;height: 55vh;"></div>
                </div>
            </div>`;

        var text_html = [];
        var entities = [];
        var entity_types = [];

        data.forEach(
            function(sentence) {

                var entity_text = ""
                var entity_type = ""

                sentence.forEach(
                    function(token) {

                         if ((entity_text != "") && ((token.prediction == 'O') || (token.prediction.startsWith('B-')))
                               || (token.prediction.slice(-3) != entity_type)) {

                            var selector = entity_text + ' ' + entity_type.slice(entity_type.length-3);

                            selector = selector.replace(/[^\w\s]|_/g, "").replace(/\s+/g, "-");

                            text_html.push(' <font color="' + getColor(entity_type) + '">'
                                                + '<a id="ent-sel-'+ entities.length +'" class="' +
                                                selector +'"> '+ entity_text + '</a>' +
                                           '</font> ');

                            entities.push(entity_text);
                            entity_types.push(entity_type);
                            entity_text = "";
                        }

                         if (token.prediction == 'O') {

                            if (text_html.length > 0) text_html.push(' ');

                            text_html.push(token.word);
                         }
                         else {
                            entity_type = token.prediction.slice(-3)

                            if (entity_text != "") entity_text += " ";

                            entity_text += token.word;
                         }
                    });

                 if ((entity_text != "") && (entity_text != null)) {

                    var selector = entity_text + ' ' + entity_type;

                    selector = selector.replace(/[^\w\s]|_/g, "").replace(/\s+/g, "-");

                    text_html.push(' <font color="' + getColor(entity_type) + '">'
                                                + '<a id="ent-sel-'+ entities.length +'"class="' +
                                                        selector +'"> '+ entity_text + '</a>' +
                                           '</font> ');

                    entities.push(entity_text);
                    entity_types.push(entity_type);
                 }

                 text_html.push('<br/>');
            }
        )
        $("#result-text").html(text_region_html);
        $("#ner-text")[0].innerHTML = text_html.join("");

        entities.forEach(
            function(entity, idx) {
                var selector = entity + ' ' + entity_types[idx];

                selector = '.' + selector.replace(/[^\w\s]|_/g, "").replace(/\s+/g, "-");

                $("#ent-sel-" + idx).click(
                    function() {
                        $(".selected").removeClass('selected');

                        selectEntity(entity + "-" + entity_types[idx],
                            function() {
                                $(selector).addClass('selected');
                            }
                        );
                    }
                );
            }
        );
    }

    that = {
        ned_url: null,
        init:
            function(ner_url, parse_url, ned_url, input_text) {

                that.ned_url = ned_url;

                $("#result-text").empty();
                $("#ner-text").empty();

                runNER(ner_url, input_text,
                    function (ner_result) {
                        showNERText(ner_result);

                        console.log(ner_result);

                        parseNER(parse_url, ner_result,
                            function(ned_result) {
                                console.log(ned_result);
                            });
                    }
                );
            }
    };

    return that;
}

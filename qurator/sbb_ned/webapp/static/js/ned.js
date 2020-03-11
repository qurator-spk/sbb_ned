
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

    function runNER (input_text, onSuccess) {

        $("#result-entities").html("");
        $("#result-text").html(spinner_html);

        let post_data = { "text" : input_text };

        $.ajax({
                url:  "../ner/ner/1" ,
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

    function parseNER (input, onSuccess) {

        let post_data = input;

        $.ajax({
                url:  "parse" ,
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

    function runNED (input, onSuccess) {

        if (ner_parsed == null) {
            console.log('Parsed NER data missing.');
            return;
        }

        let post_data = input;

        $.ajax({
                url:  "ned" ,
                data: JSON.stringify(post_data),
                type: 'POST',
                contentType: "application/json",
                success:
                    function(result) {
                        Object.assign(ned_result, result);
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

    function makeResultList(entities) {
        var entities_html = "";

        entities.forEach(
            function(candidate, index) {

                if (index > 10) return;

                //if (Number(candidate[1]) < 0.1) return;

                entities_html += '<a href="https://de.wikipedia.org/wiki/' + candidate[0] + '">'
                                        + candidate[0] + '</a> '
                                        + '(' + Number(candidate[1]['sentence_score']).toFixed(2)
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
            makeResultList(ned_result[entity]);
            onSuccess();
            return;
        }

        if (!(entity in ner_parsed) ){
            $("#result-entities").html("NO NER DATA.");
            return;
        }

        var input = {};
        input[entity] = ner_parsed[entity];

        runNED(input,
            function() {
                if (entity in ned_result) {
                    makeResultList(ned_result[entity]);

                    onSuccess();
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

        var text_html = "";
        var entities = [];
        var entity_types = [];

        data.forEach(
            function(sentence) {

                var entity_text = ""
                var entity_type = ""

                sentence.forEach(
                    function(token) {

                         if ((entity_text != "") && ((token.prediction == 'O') || (token.prediction.startsWith('B-')))) {

                            var selector = entity_text + ' ' + entity_type.slice(entity_type.length-3);

                            selector = selector.replace(/[^\w\s]|_/g, "").replace(/\s+/g, "-");

                            console.log('HTML:', entity_text, entity_type, selector);

                            text_html += ' <font color="' + getColor(entity_type) + '">'
                                                + '<a id="ent-sel-'+ entities.length +'" class="' +
                                                selector +'"> '+ entity_text + '</a>' +
                                           '</font> ';

                            entities.push(entity_text);
                            entity_types.push(entity_type.slice(entity_type.length-3));
                            entity_text = "";
                        }

                         if (token.prediction == 'O') {

                            if (text_html != "") text_html += ' ';

                            text_html += token.word;
                         }
                         else {
                            entity_type = token.prediction

                            if (entity_text != "") entity_text += " ";

                            entity_text += token.word;
                         }
                    });

                 if ((entity_text != "") && (entity_text != null)) {

                    var selector = entity_text + ' ' + entity_type.slice(entity_type.length-3);

                    console.log('HTML:', entity_text, entity_type, selector);

                    selector = selector.replace(/[^\w\s]|_/g, "").replace(/\s+/g, "-");

                    text_html += ' <font color="' + getColor(entity_type) + '">'
                                                + '<a id="ent-sel-'+ entities.length +'"class="' +
                                                        selector +'"> '+ entity_text + '</a>' +
                                           '</font> ';

                    entities.push(entity_text);
                    entity_types.push(entity_type.slice(entity_type.length-3));
                 }

                 text_html += '<br/>';
            }
        )
        $("#result-text").html(text_region_html);
        $("#ner-text").html(text_html);

        entities.forEach(
            function(entity, idx) {
                var selector = entity + ' ' + entity_types[idx];

                selector = '.' + selector.replace(/[^\w\s]|_/g, "").replace(/\s+/g, "-");

                console.log('Function:' , selector);

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
        init:
            function(input_text) {

                $("#result-text").empty();
                $("#ner-text").empty();

                runNER(input_text,
                    function (ner_result) {
                        showNERText(ner_result);

                        console.log(ner_result);

                        parseNER(ner_result,
                            function(ned_result) {
                                console.log(ned_result);
                            });
                    }
                );
            }
    };

    return that;
}

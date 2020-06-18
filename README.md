
![sbb-ner-demo example](.screenshots/sbb_ned_demo.png?raw=true)

***

# Installation:

Before named entity disambiguation (NED) can be performed, 
the input text has to be NER-tagged.
Our NED system provides a HTTP-interface 
that accepts the NER-tagged input in JSON format. 

In order to try our NED-system, 
you can either use some NER-tagger 
and convert the output of that system into the expected format, 
or you can download and install the [SBB-NER-tagger](../sbb_ner) 
and use the output of that system
as input of our NED-system.

Please read the installation guide of the [SBB-NER-tagger](../sbb_ner) 
for more detailed information about the expected input format of the NED-system.

If you want to use the NED - demo web interface as it is shown in the image above,
you have to 
* install and run the [SBB-NER-tagger](../sbb_ner)
* install and run the SBB-NED system
* setup an nginx installation (or other HTTP proxy) such that the NER and the NED system are available 
behind a URL-structure as it is defined by the nginx configuration example below:
```
server {
    listen 80 default_server;
    server_name _;

    client_max_body_size 2048M;

    location /sbb-tools/ner/ {
     proxy_pass http://localhost:5000/;
     proxy_connect_timeout       360000s;
     proxy_send_timeout          360000s;
     proxy_read_timeout          360000s;
     send_timeout                360000s;
    }

    location /sbb-tools/ned/ {
     proxy_pass http://localhost:5001/;
     proxy_connect_timeout       360000s;
     proxy_send_timeout          360000s;
     proxy_read_timeout          360000s;
     send_timeout                360000s;
    }
}
```

NED web-interface is availabe at http://localhost/sbb-tools/ned/index.html . 

NED as it is done by our system is computationally demanding, 
therefore computations in particular for larger documents can take a long time.
Therefore the nginx configuration contains the very high timeout settings for proxy 
connections since otherwise the connection could break before the result of the 
computation has been submitted.

Note: If there is another proxy in between, the connection can break due to timeouts
within that proxy! HTTP obviously is not made to perform single requests of long 
durations, therefore we recommend to split up processing of larger documents in smaller
requests which is possible due to the design of our system. 
However, for academic purposes it sometimes is more convenient to do large requests 
where the computation might take several hours.

***

## Installation of the NED-core:

Setup virtual environment:
```
virtualenv --python=python3.6 venv
```

Activate virtual environment:
```
source venv/bin/activate
```

Upgrade pip:
```
pip install -U pip
```

Install package together with its dependencies in development mode:
```
pip install -e ./
```

Download required models: https://qurator-data.de/sbb_ned/models.tar.gz 

Beware: The archive file contains the required models as well as the knowledge bases
for german, french and english, altogether roughly 200GB!!! 

Extract model archive:
```
tar -xzf models.tar.gz
```

Run webapp directly:

```
env CONFIG=de-config.json env FLASK_APP=qurator/sbb_ned/webapp/app.py env FLASK_ENV=development env USE_CUDA=True flask run --host=0.0.0.0 --port=5001
```
Replace de-config.json by fr-config.json or en-config.json to switch to french or english.
Set USE_CUDA=False, if you do not have a GPU available/installed 
(This NED already takes some time with GPU, it might not be feasible without GPU).

***


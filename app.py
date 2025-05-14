from flask import Flask,request,jsonify,stream_with_context
import os
from termcolor import colored
import shutil
import re
from utils import CodeReviewerBot

app = Flask(__name__)


@app.route('/file_save',methods=['POST'])
@cross_origin()
def File_Save():
    shutil.rmtree('/data', ignore_errors=True)
    for i in os.listdir('../database'):
        os.remove(os.path.join('../database',i))
    ls = os.listdir('../database')
    content = request.files
    a = 'File Not Uploaded'
    temp = []
    if content:
        for file in content:
            filelist = content.getlist(f'{file}')
            for file1 in filelist:
                filename = os.path.basename(file1.filename)
                file1.save(os.path.join('../database',filename))
                a = 'File Uploaded'
                unsupported_types = ['docx','xlsx','pptx']
                pattern = r'\.(.*)$'
                match = re.search(pattern, filename)
                if  match.group(1) in unsupported_types:
                    return "FILE NOT SUPPORTED"

    return jsonify({'status':a})

@app.route('/agent_query',methods=['GET'])
@cross_origin()
def Agent_Query():
    rev_bot = Bot('../database')
    print(rev)
    print(response)
    return jsonify({'review':rev,'unittest':response})


if __name__ == '__main__':
    app.run(port=8080,debug=True)

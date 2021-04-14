import os
from flask import Flask, render_template, request, redirect, url_for,send_file
from werkzeug import secure_filename
import time
import uuid
import base64
import pandas as pd
import re
from PIL import Image
import numpy as np
import ast
from azure.core.exceptions import ResourceNotFoundError
from azure.ai.formrecognizer import FormRecognizerClient
from azure.ai.formrecognizer import FormTrainingClient
from azure.core.credentials import AzureKeyCredential
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import FormRecognizerClient
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import time
import cv2
from operator import itemgetter

# Replace with valid values
ENDPOINT = "https://quixycustomvision.cognitiveservices.azure.com/"
training_key = "ae497ae143ba4ebcb669bfde53b27ab1"
prediction_key = "c9b8ade786584e809ea130d49468aca3"
prediction_resource_id = "/subscriptions/07f1736a-4d16-445f-8286-ecbda63807ae/resourceGroups/MLSamples/providers/Microsoft.CognitiveServices/accounts/QuixyCustomVision"

credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

project_id = "044b0536-e3f0-4132-b785-be7166dab1c4"
publish_iteration_name = "Iteration11"


df=pd.DataFrame(columns=['SectionName','ElementType','LabelName','FieldName','HelpText','Data','DefaultValue','IsHiddenField','Required','x1','x0'])
inter = pd.DataFrame(columns=['File_Name','Text','bounding_box'])
File_Name=[]
Text=[]
bounding_box=[]



endpoint = "https://quixyocr.cognitiveservices.azure.com/"
key = "03462e450efa4e338d200900d30d8c7b"
#key_t = "2f4b76eff9264512aa93c4f704d1ddd9"
#endpoint_t = "https://quixyresumeanalytics.cognitiveservices.azure.com/"


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'PNG', 'bmp','JPG','JPEG','png'])



        
form_recognizer_client = FormRecognizerClient(endpoint, AzureKeyCredential(key))

form_recognizer_client = FormRecognizerClient(endpoint=endpoint, credential=AzureKeyCredential(key))

def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.
    
def Point(x,y):
    q = 'x='+str(x)
    w = 'y='+str(y)
    return (q,w)
    
def cleaning(a):
    st = str(a)
    prc_a = str(a).replace("'",'')
    lst_a = re.findall(r'\(.*?\)', prc_a)
    lst = []
    for i in range(len(lst_a)):
        m = i+1
        n = lst_a[i].replace('x', 'x').replace('y', 'y')
        lst.append(n)
        print(n)

    return lst
  


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
    return render_template('template.html', label='', imagesource='../uploads/template.jpeg')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    
    global File_Name 
    global Text
    global bounding_box
    global df
    
    window_name = 'Image'

    count=0
    obj_tag = []
    obj_bbox = []
    text_tag = []
    text_bbox = []
    place_holder={}
    Left_Label={}
    top_label={}
    look_up_label={}
    look_up_element={}
    LabelName=[]
    ElementType=[]
    FieldName=[]
    HelpText=[]
    look_up_label_text={}
    radio_data=[]
    radio_label=''
    radio_element=[]
    checkbox_data=[]
    checkbox_element=[]
    if request.method == 'POST':
        
        import time
        start_time = time.time()
        files = request.files.getlist('file[]')
  
        for file in files:
            filename_ = my_random_string(6)
            bounding_box=[]
            sentence=[]
            if file and allowed_file(file.filename):
                if(1):
                    filename = secure_filename(file.filename)
                
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    with open(file_path, "rb") as f:
                        poller = form_recognizer_client.begin_recognize_content(form=f)
                    form_pages = poller.result()
                    
                    inter.at[count,'File_Name']=file.filename
                    img = Image.open(file.filename)
                    image_cv = cv2.imread(file.filename)
                    width,height = img.size
                    print(width)
                    print(height)
                    
                    for idx, content in enumerate(form_pages):
                                #print("----Recognizing content from page #{}----".format(idx+1))
                                #print(content.lines)
                                
                                for line_idx, line in enumerate(content.lines):
                                    print(line.text)
                                    v = line.text
                                    text_tag.append(v)
                                    
                                    #print(bounding_box)
                                    #print(line.bounding_box[0])
                                    #print(line.bounding_box[1])
                                    #print(line.bounding_box[2])
                                    #print(line.bounding_box[3])
                                    inter.at[count,'Text']=line.text
                                    bounding_box.append(cleaning(line.bounding_box))
                                    inter.at[count,'bounding_box']=cleaning(line.bounding_box)
            
                                    File_Name=[]
                                    Text=[]
                                    
                                    count=count+1
                    print(bounding_box)
                    for i in bounding_box:
                        temp = re.findall(r'\d+', str(i))
                        res = list(map(int, temp))
                        text_lst = res[0],res[2],res[8],res[10]
                        text_bbox.append(text_lst)
                        #print(res[0],res[2],res[8],res[10])
                        start_point = (int(res[0]), int(res[2]))
                        end_point = ( int(res[8]),int(res[10]))
                        # Blue color in BGR 
                        color = (255, 0, 0) 
                        
                        # Line thickness of 2 px 
                        thickness = 2
                        
                        # Using cv2.rectangle() method 
                        # Draw a rectangle with blue line borders of thickness of 2 px 
                        img = cv2.rectangle(image_cv, start_point, end_point, color, thickness) 
                        
                        # Displaying the image 
                        path = 'C:/Resume_Parser/ObjectDetection/static'
                        cv2.imwrite(os.path.join(path , 'waka.jpg'), img)
                        
                                
                        
                    with open( file.filename, "rb") as image_contents:
                        results = predictor.detect_image(project_id, publish_iteration_name, image_contents.read())
                        # Display the results.
                        i=0
                        for prediction in results.predictions:
                            if((prediction.probability * 100)>18):
                                print("\t" + prediction.tag_name + ": {0:.2f}%".format(prediction.probability * 100))
                                tag = prediction.tag_name
                                a= str(prediction.bounding_box)
                                a = eval(a)
                                obj_tag.append(tag)
                                #print(a)
                                l = a['left']*width
                                t = a['top']*height
                                w = a['width']*width
                                h = a['height']*height
                                lst = [l, t, w+l, h+t]
                                #print(lst)
                                obj_bbox.append(lst)
                                
                    
                                # Start coordinate, here (5, 5) 
                                # represents the top left corner of rectangle 
                                #start_point = (5, 5) 
                                start_point = (int(l), int(t))
                                
                                # Ending coordinate, here (220, 220) 
                                # represents the bottom right corner of rectangle 
                                #end_point = (220, 220) 
                                end_point = ( int(w+l),int(h+t))
                                
                                # Blue color in BGR 
                                color = ((255, 51, 0),(255, 102, 0),(0, 255, 0)) 
                                
                                # Line thickness of 2 px 
                                thickness = 2
                                print(start_point,end_point)
                                # Using cv2.rectangle() method 
                                # Draw a rectangle with blue line borders of thickness of 2 px 
                                img = cv2.rectangle(image_cv, start_point, end_point, color[i], thickness) 
                                
                                filename1 = '/static/final.jpg'
                                # Displaying the image  
                                
                                path = 'C:/Resume_Parser/ObjectDetection/static'
                                cv2.imwrite(os.path.join(path , filename_+'.jpg'), img) 
                                
            
            bounding_box=[]                   
            
        l=0
        
        print(text_bbox)
        print(obj_bbox)
        print(text_tag)
        
        
        #obj_bbox=sorted(obj_bbox, key=itemgetter(1))
        #text_bbox=sorted(text_bbox, key=itemgetter(1))
        
        for i in text_bbox: 
            look_up_label_text[str(i)] = text_tag[text_bbox.index(i)]
        
        for i in text_bbox: 
            look_up_label[str(i)] = None
        for i in obj_bbox: 
            look_up_element[str(i)] = None
            
            
        for i in obj_bbox:
            temp=[[0,0,0,0]]
            
            if(obj_tag[obj_bbox.index(i)]=='Text Box' or obj_tag[obj_bbox.index(i)]=='Text Area'or obj_tag[obj_bbox.index(i)]=='Dropdown' or obj_tag[obj_bbox.index(i)]=='Date' ):
            #Within
                for j in text_bbox:
                    
                    if(look_up_label[str(j)]!=1):
                    
                        
                        if(i[0]<=j[0] and i[1]<=j[1] and i[2]>=j[2] and i[3]>=j[3]):
                            place_holder[str(i)] = look_up_label_text[str(j)]
                            #inter.append(labels[list4.index(j)]+' '+'within')
                            look_up_label[str(j)]=1
                            HelpText.append(look_up_label_text[str(j)])
                            #ElementType.append(obj_tag[obj_bbox.index(i)])
                            if(obj_tag[obj_bbox.index(i)]=='Button'):
                                look_up_element[str(i)]=1
                #Left           
                for j in text_bbox:
                    if(look_up_label[str(j)]!=1  and look_up_element[str(i)]!=1):
                        if(i[0]>=j[0] and i[1]<=j[1] and i[2]>=j[2] and i[3]>=j[3]):
                            Left_Label[str(i)] = look_up_label_text[str(j)]
                            #inter.append(labels[list4.index(j)]+' '+'left')
                            look_up_label[str(j)]=1
                            look_up_element[str(i)]=1
                            LabelName.append(look_up_label_text[str(j)])
                            ElementType.append(obj_tag[obj_bbox.index(i)])
                            FieldName.append(look_up_label_text[str(j)])
                #Top            
                for j in text_bbox:
                    if(look_up_label[str(j)]!=1 and look_up_element[str(i)]!=1):
                        #Top
                        if(i[1]>=j[1] and i[3]>=j[3]  and i[2]>=j[2]):
                            
                            if(j[1]>=temp[0][1] and (i[0]<=j[2])):
                                temp[0]=j
                                c=j
                                #inter.append(labels[list4.index(temp[0])]+' '+'top')
                                
                        
                    
                            
                if(temp[0]!=[0,0,0,0]): 
                    print(look_up_label_text)
                    print(temp[0])
                    top_label[str(i)]=look_up_label_text[str(c)]
                    LabelName.append(look_up_label_text[str(c)])
                    ElementType.append(obj_tag[obj_bbox.index(i)])
                    FieldName.append(look_up_label_text[str(c)])
                    look_up_label[str(c)]=1
                    look_up_element[str(i)]=1
                  
                if(LabelName):
                    df.at[l,'LabelName']=' '.join(map(str, LabelName))
                else:
                    df.at[l,'LabelName']=np.nan
                if(FieldName):
                    df.at[l,'FieldName']=' '.join(map(str, FieldName))
                else:
                    df.at[l,'FieldName']=np.nan
                if(HelpText):
                    df.at[l,'HelpText']=' '.join(map(str, HelpText))
                else:
                    df.at[l,'HelpText']=np.nan
                if(ElementType):
                    df.at[l,'ElementType']=' '.join(map(str, ElementType))
                    df.at[l,'x1']=i[1]
                    df.at[l,'x0']=i[0]
                else:
                    df.at[l,'ElementType']=np.nan
                df.at[l,'SectionName']='Section1'
                if(obj_tag[obj_bbox.index(i)]=='Dropdown'):
                    df.at[l,'Data']=' '.join(map(str, HelpText))
                df.at[l,'DefaultValue']='ABC'
                df.at[l,'IsHiddenField']='FALSE'
                df.at[l,'Required']='FALSE'
            
                #ele[str(i)]=inter
                #inter=[]
                l=l+1
                LabelName=[]
                ElementType=[]
                FieldName=[]
                HelpText=[]
                df['ElementType'][df['ElementType']=='Text Box']='TextBox'
                df['ElementType'][df['ElementType']=='Dropdown']='DropDown'
                df['ElementType'][df['ElementType']=='Text Area']='TextArea'
                
                
            for j in text_bbox:
                
                
                if(obj_tag[obj_bbox.index(i)]=='Radio Button'):
                    
                    
                    if((look_up_label[str(j)]!=1 and look_up_element[str(i)]!=1) and (len(look_up_label_text[str(j)])>1)):
                        
                        
                        if((i[2]<=j[2]) and ((j[1] in range(int(i[1]+1),int(i[3]+1))) or (j[1] in range(int(i[1]-1),int(i[3]-1))) or (j[3] in range(int(i[1]+1),int(i[3]+1))) or (i[1] in range(int(j[1]+1),int(j[3]+1))) or (i[3] in range(int(j[1]+1),int(j[3]+1))) )):
                            
                            #Right
                            
                            radio_data.append(look_up_label_text[str(j)])
                            radio_element.append(i)
                            look_up_label[str(j)]=1
                            look_up_element[str(i)]=1   
                            
                            
                            
                if(obj_tag[obj_bbox.index(i)]=='Check Box'):
                    
                    
                    if((look_up_label[str(j)]!=1 and look_up_element[str(i)]!=1) and (len(look_up_label_text[str(j)])>1)):
                        
                        
                        if((i[2]<=j[2]) and ((j[1] in range(int(i[1]+1),int(i[3]+1))) or (j[1] in range(int(i[1]-1),int(i[3]-1))) or (j[3] in range(int(i[1]+1),int(i[3]+1))) or (i[1] in range(int(j[1]+1),int(j[3]+1))) or (i[3] in range(int(j[1]+1),int(j[3]+1))) )):
                            
                            #Right
                            
                            checkbox_data.append(look_up_label_text[str(j)])
                            checkbox_element.append(i)
                            look_up_label[str(j)]=1
                            look_up_element[str(i)]=1 
        print("------------------------------------------>",radio_data)
        print("------------------------------------------>",radio_element)
        temp1=[] 
        indx=0
        
        radio_data_cleaned=[]
        for text_radio in radio_data:
            
            inter_text=re.sub("^o |^O ", "", text_radio)
            final_text=re.sub(" o | O | 0 ", ",", inter_text)
            
            radio_data_cleaned.append(final_text)
            
        checkbox_data_cleaned=[]
        for text_check in checkbox_data:
            
            inter_text=re.sub("^o |^O ", "", text_check)
            checkbox_data_cleaned.append(inter_text)
            
        
        
            
        
        s=sorted(radio_element,key=itemgetter(1))
        
        print("SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS",s)
        d={}
        
        l=len(s)
        for i in range(0,l):
            c=[]
            d[i]=c
            for j in range(0,l):
                if(abs(s[i][1]-s[j][1])<=1):
                    c.append(s[j])
        result={}
        i=0
        for key,value in d.items():
            if value not in result.values():
                result[i] = value
                i=i+1
        
        
        ans={}
        for key,val in result.items():
            v=sorted(val, key=itemgetter(0))
            ans[key]=v
        
        final_list=[]
        
        print("annsssssssssssssssssssssssssssss",ans)
        
        for i in ans:
            final_list.append(ans[i])
        
        print(final_list)
        from functools import reduce #python 3
        
        if(final_list):
            radio_element1=reduce(lambda x,y: x+y,final_list)
        
            sorted_radio_elements=[]
        
            for i in radio_element1:
                sorted_radio_elements.append(radio_data_cleaned[radio_element.index(i)])
        
            print("------------------------------------------ After >",radio_element)
            print("---------------------------------",sorted_radio_elements)
        
            for idx,i in enumerate(radio_element1):
            
                for j in text_bbox:
                
                    match=re.search('^O ', look_up_label_text[str(j)])
                    if match is None:
                        matches=1
                    else:
                        matches=0
                    if((look_up_label[str(j)]!=1 and (len(look_up_label_text[str(j)])>1) and matches)):
                        if(j[0]<i[0] and ((j[1] in range(int(i[1]+1),int(i[3]+1))) or (j[1] in range(int(i[1]-1),int(i[3]-1))) or (j[3] in range(int(i[1]+1),int(i[3]+1))) or (i[1] in range(int(j[1]+1),int(j[3]+1))) or (i[3] in range(int(j[1]+1),int(j[3]+1))) )):
                            temp1=j
                        
                        
                        if(i[1]>=j[1]):
                            temp1=j
                        
            
            
                if(temp1):
                    print("Entered")
                    df.at[l,'LabelName']=text_tag[text_bbox.index(temp1)]
                    df.at[l,'FieldName']=text_tag[text_bbox.index(temp1)]
                    df.at[l,'Data']= sorted_radio_elements[idx]
                    df.at[l,'HelpText']= sorted_radio_elements[idx]
                    if(obj_tag[obj_bbox.index(i)]=='Radio Button'):
                        df.at[l,'ElementType']='Radio'
                        df.at[l,'x1']=temp1[1]
                        df.at[l,'x0']=temp1[0]
                    df.at[l,'SectionName']='Section1'            
                    df.at[l,'DefaultValue']='ABC'
                    df.at[l,'IsHiddenField']='FALSE'
                    df.at[l,'Required']='FALSE'
                    l=l+1  
                    temp1=[]
                
                
        
        for cidx,i in enumerate(checkbox_data_cleaned):
            
            df.at[l,'LabelName']=i
            df.at[l,'FieldName']=i
            df.at[l,'x1']=checkbox_element[cidx][1]
            df.at[l,'x0']=checkbox_element[cidx][0]
            df.at[l,'ElementType']='CheckBox'
            df.at[l,'SectionName']='Section1'            
            df.at[l,'DefaultValue']='ABC'
            df.at[l,'IsHiddenField']='FALSE'
            df.at[l,'Required']='FALSE'
            l=l+1  
             
            
            
        print(df)
        df1=df[(df['ElementType']=='Radio')]
        if(len(df1.index) != 0):
            print("Here--------------------------------------------------------------------")
            df1=df1.groupby(['SectionName','ElementType','LabelName','FieldName','DefaultValue','IsHiddenField', 'Required','x1','x0'])['Data','HelpText'].agg(', '.join).reset_index()
            print(df1)
            df.drop(df[df.ElementType=='Radio'].index, inplace=True)
            
            
            df2=pd.concat([df,df1])
            print(df2)
            df2=df2.sort_values(by=['x1', 'x0'])
            print(df2)
            df2=df2.drop(['x1','x0'],axis=1)
            df2.dropna(axis=0, thresh =6,inplace=True)
            pre_pros_text=[]
            for i in (df2['LabelName']):
                if(i):
                    
                    pre_pros_text.append(re.sub('[^A-Za-z0-9]+', ' ', i))
            df2['LabelName']=pre_pros_text
            pre_pros_text=[]
            for i in (df2['FieldName']):
                if(i):
                    pre_pros_text.append(re.sub('[^A-Za-z0-9]+', ' ', i))
            df2['FieldName']=pre_pros_text
            print(df2)
            df2.to_excel('uploads/'+filename_+'.xlsx',index=False)
            df2=df2.iloc[0:0,:]
            df1=df1.iloc[0:0,:]
            df=df.iloc[0:0,:]
        else:
            print("Hey%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            df.dropna(axis=0, thresh =6,inplace=True)
            print(df)
            df=df.sort_values(by=['x1', 'x0'])
            df=df.drop(['x1','x0'],axis=1)
            print(df)
            pre_pros_text=[]
            for i in (df['LabelName']):
                if(i):
                    
                    pre_pros_text.append(re.sub('[^A-Za-z0-9]+', ' ', i))
            df['LabelName']=pre_pros_text
            pre_pros_text=[]
            for i in (df['FieldName']):
                if(i):
                    pre_pros_text.append(re.sub('[^A-Za-z0-9]+', ' ', i))
            df['FieldName']=pre_pros_text
            print(df)
            df.to_excel('uploads/'+filename_+'.xlsx',index=False)
            df=df.iloc[0:0,:]
   
        
        
        
        
        count=0
        obj_tag = []
        obj_bbox = []
        text_tag = []
        text_bbox = []
        place_holder={}
        Left_Label={}
        top_label={}
        look_up_label={}
        look_up_element={}
        LabelName=[]
        ElementType=[]
        FieldName=[]
        HelpText=[]
        look_up_label_text={}
        radio_data=[]
        radio_label=''
        radio_element=[]
        checkbox_data=[]
        checkbox_element=[]

    return render_template('template.html',filename=filename_+'.xlsx')


from flask import send_from_directory

@app.route('/uploads/<filename>')
def database_download(filename):
    return send_from_directory('database_reports', filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

from werkzeug import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})


if __name__ == "__main__":
    app.debug=False
    app.run(host='0.0.0.0', port=3000)
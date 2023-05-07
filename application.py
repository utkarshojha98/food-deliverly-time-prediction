from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

application=Flask(__name__)
app=application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            Delivery_person_Age       =float(request.form.get('Delivery_person_Age')),      
            Delivery_person_Ratings      =float(request.form.get('Delivery_person_Ratings')) ,     
            Restaurant_latitude     =float(request.form.get('Restaurant_latitude')),   
            Restaurant_longitude           =float(request.form.get('Restaurant_longitude')) ,        
            Delivery_location_latitude     =float(request.form.get('Delivery_location_latitude')),    
            Delivery_location_longitude    =float(request.form.get('Delivery_location_longitude') ),   
            day                      =float(request.form.get('day') ) ,                    
            month                      =float(request.form.get('month') ) ,                    
            year                      =float(request.form.get('year') ) ,                    
            hour_Order_picked                      =float(request.form.get('hour_Order_picked') ) ,                    
            min_Order_picked                      =float(request.form.get('min_Order_picked') ) ,                    
            order_hour                      =float(request.form.get('order_hour') ) ,                    
            order_min                      =float(request.form.get('order_min') ) ,          
            Weather_conditions              =request.form.get('Weather_conditions'),           
            Road_traffic_density            =request.form.get('Road_traffic_density'),          
            Vehicle_condition                =request.form.get('Vehicle_condition'),               
            Type_of_order                   =request.form.get('Type_of_order') ,             
            Type_of_vehicle                 =request.form.get('Type_of_vehicle'),                
            multiple_deliveries            =request.form.get('multiple_deliveries'),            
            Festival                        =request.form.get('Festival'),                    
            City                            =request.form.get('City')        
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('results.html',final_result=results)






if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)
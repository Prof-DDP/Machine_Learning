# Function I make for models that could be useful in the future

class Multiple_Linear_Regression:
      
    def y_eval(y1, y2, og):
        import numpy as np
        total = len(og)
        y1_score = 0
        y1_correct = 0
                
        y2_score = 0
        y2_correct = 0
                
        for i in range(total):
            y1_diff = np.abs(y1[i] - og[i])
            y2_diff = np.abs(y2[i] - og[i])
                    
            if y1_diff > y2_diff:
                y2_score+=1
                if y2_diff == 0:
                    y2_correct+=1
                            
            elif y2_diff > y1_diff:
                y1_score+=1
                
                if y1_diff == 0:
                    y1_correct+=1
                    
            else:
                y2_score+=1
                y1_score+=1
                        
                if y2_diff == 0:
                    y2_correct+=1
                        
                if y1_diff == 0:
                    y1_correct+=1
                        
                
        y1_score_percent = (y1_score / (y1_score + y2_score)) * 100
        y1_correct_percent = (y1_correct / total) * 100
                
        y2_score_percent = (y2_score / (y1_score + y2_score)) * 100
        y2_correct_percent = (y2_correct / total) * 100
                
        print_statement = '{} was the more accurate predictor!'
        print_statement2 = 'y1 Score Percentage: {}  vs.  y2 Score Percentage: {}'
        print_statement3 = '{} was {}% more accurate'
        print_statement4 = '{} had {}% exact accuracy'
                
        print(print_statement2.format(y1_score_percent, y2_score_percent))
                
        if y1_score > y2_score:
            print(print_statement.format('y1'))
            percent_diff = y1_score_percent - y2_score_percent
            print(print_statement3.format('y1', percent_diff))
            print(print_statement4.format('y1', y1_correct_percent))
                    
        elif y2_score > y1_score:
            print(print_statement.format('y2'))
            percent_diff = y2_score_percent - y1_score_percent
            print(print_statement3.format('y2', percent_diff))
            print(print_statement4.format('y2', y2_correct_percent))
                    
        else:
            print('y1 and y2 are both equally accurate.')




            
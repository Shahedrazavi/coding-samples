#!/usr/bin/env python
# coding: utf-8

# # AI Fall 2022 - A4 - Regression
# ## Your info
# 
# **Student Name:** _Shahed Razavi_
# 
# **Student Id:** _99104627_

# #### Allowed packages: Pandas, matplotlib, seaborn, and numpy. Sklearn is allowed only for getting the dataset.

# In[212]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split
from helper import *


# # some data processing and feature analysis
# 
# <li> load boston house dataset </li>
# <li> split train and test with ratio 1 to 3 </li>
# <li> plot the target value based on 13 different features and recognize the correlation between features and
# the target values. talk about them and their meanings.</li>

# In[213]:


data,X,y = get_data_normalized()
X_train, X_test, y_train, y_test = split_data(X,y,0.25)


# In[214]:


plt.scatter(X_test.CRIM,y_test)
plt.xlabel("CRIM")
plt.ylabel("Target Value")
plt.show()
plt.scatter(X_test.ZN,y_test)
plt.xlabel("ZN")
plt.ylabel("Target Value")
plt.show()
plt.scatter(X_test.INDUS,y_test)
plt.xlabel("INDUS")
plt.ylabel("Target Value")
plt.show()
plt.scatter(X_test.CHAS,y_test)
plt.xlabel("CHAS")
plt.ylabel("Target Value")
plt.show()
plt.scatter(X_test.NOX,y_test)
plt.xlabel("NOX")
plt.ylabel("Target Value")
plt.show()
plt.scatter(X_test.RM,y_test)
plt.xlabel("RM")
plt.ylabel("Target Value")
plt.show()
plt.scatter(X_test.AGE,y_test)
plt.xlabel("AGE")
plt.ylabel("Target Value")
plt.show()
plt.scatter(X_test.DIS,y_test)
plt.xlabel("DIS")
plt.ylabel("Target Value")
plt.show()
plt.scatter(X_test.RAD,y_test)
plt.xlabel("RAD")
plt.ylabel("Target Value")
plt.show()
plt.scatter(X_test.TAX,y_test)
plt.xlabel("TAX")
plt.ylabel("Target Value")
plt.show()
plt.scatter(X_test.PTRATIO,y_test)
plt.xlabel("PTRATIO")
plt.ylabel("Target Value")
plt.show()
plt.scatter(X_test.B,y_test)
plt.xlabel("B")
plt.ylabel("Target Value")
plt.show()
plt.scatter(X_test.LSTAT,y_test)
plt.xlabel("LSTAT")
plt.ylabel("Target Value")
plt.show()


# در نمودارهای قبلی مشاهده می‌کنید که رابطه متغیر هدف با بعضی از پارامترهای توضیحی، یک رابطه خوب تابعی، حال یا خطی یا لگاریتمی یا درجه دو است و رابطه بعضی از پارامترها با متغیر هدف به این صورت نیست، زیرا بعضی از پارامترها متغیرهای گسسته هستند و یا مشاهده می‌کنیم که به ازای یک سری مقادیر خاص از یک پارامتر، داده‌های زیادی با مقدار متغیر هدف‌های متفاوت در آن جای نمودار قرار گرفته‌اند.

# ## close form of Linear Regression
# Minimize 
# $$
# \frac{1}{2} (Y-\phi W)^T(Y-\phi W) + \frac{1}{2} \lambda W^TW
# $$
# <li> 1-write down close form of linear regression </li>
# <li> 2-now use this close form to obtain good weight for this problem </li>
# <li> 3-Plot the target value and the predicted value based on ‘LSTAT’, ‘DIS’, and any other
# features so that you can see how the distributions vary</li>
# <li> 4-plot regularization - weights_norm with lambda between 0 to 0.1 with step 0.005 </li>
# <li> 5-plot regularization - test_error with lambda between 0 to 0.1 with step 0.005 </li>
# <li> 6-explain effect of regularization </li>
# <li> 7-add square of each feature to data set and repeat 4,5,6</li>
# <li> 8-add square and power of three of each feature to data set and repeat 4,5,6</li>
# <li> compare part $7^{th}$ test error and previous one <b>explain how 7 and 8 are helping model to do better work </li>
#     
# 

# 1. The Close form of regression is:
# $w = (\phi^T \phi+\lambda n I)^{-1} \phi^T y$

# In[215]:


#this is just template you are free to implement is however you want. add many cell as you wish
class LinearRegressionModel: #phi is phi = lambda X : np.c_[np.ones(X.shape[0]),X] for adding bias term to data or 
    # any other features to data (this is just suggestion you are free to do whatever you want.)
    def __init__(self):
        self.phi = lambda X : np.c_[np.ones(X.shape[0]),X]
        self.w = []
        pass
    

    def fit(self,X,y,regularization):
        """
        get X and y train and learn the parameter by the equation.
        
        """
        self.w = np.linalg.inv(self.phi(X).T @ self.phi(X) + regularization * np.identity(X.shape[1]+1)) @ self.phi(X).T @ y
        
    def evaluate(self,X,y):
        """
        get X and y and calculate error.
        """
        r = ((y - self.phi(X) @ self.w).T @ (y - self.phi(X) @ self.w))/X.shape[0]
        return r
    
    def transform(self,X):
        """
        get X and calculate yhat as predicted values.
    
        """
        return self.phi(X) @ self.w
    
    def get_param(self):
        "return w "
        return self.w


# ### 3:

# In[216]:


model = LinearRegressionModel()
model.fit(X_train,y_train,0)
yhat = model.transform(X_test)

plt.scatter(X_test.LSTAT, y_test, label='Target Value')
plt.scatter(X_test.LSTAT, yhat, label ='Predictions')
plt.xlabel("LSTAT")
plt.legend();
plt.show()


# In[217]:


plt.scatter(X_test.DIS, y_test, label='Target Value')
plt.scatter(X_test.DIS, yhat, label ='Predictions')
plt.xlabel("DIS")
plt.legend();
plt.show()


# In[218]:


plt.scatter(X_test.AGE, y_test, label='Target Value')
plt.scatter(X_test.AGE, yhat,label ='Predictions')
plt.xlabel("Age")
plt.legend();
plt.show()


# In[219]:


plt.scatter(X_test.CRIM, y_test, label='Target Value')
plt.scatter(X_test.CRIM, yhat,label ='Predictions')
plt.xlabel("CRIM")
plt.legend();
plt.show()


# In[220]:


plt.scatter(X_test.PTRATIO, y_test, label='First')
plt.scatter(X_test.PTRATIO, yhat,label ='Second')
plt.xlabel("PTRATIO")
plt.legend();
plt.show()


# ### 4:

# In[221]:


x = [i/200 for i in range(21)]
y = []
for lam in x:
    model.fit(X_train,y_train,lam)
    w = model.get_param()
    y.append(np.sqrt(w.T @ w))
    
plt.plot(x,y);
plt.xlabel("Lambda")
plt.ylabel("Weights Norm")
plt.show()


# ### 5:

# In[222]:


x = [i/200 for i in range(21)]
y = []
for lam in x:
    model.fit(X_train,y_train,lam)
    w = model.evaluate(X_test, y_test)
    y.append(w)
    
plt.plot(x,y);
plt.xlabel("Lambda")
plt.ylabel("MSE")
plt.show()


# ### 6:
# عبارت رگولاریزیشن در عبارت هدف مسئله بهینه سازی باعث می‌شود که به نوعی یک پنالتی به ازای مقادیری که در آن مقادیر درایه‌های
# دبلیو
# بزرگ می‌شود ایجاد می‌کند تا به این ترتیب جواب مسئله بهینه سازی به ازای حالتی رخ بدهد که تا حد ممکن ضرایب
# دبلیو
# کوچکتر بشوند. و هدف ما از انجام این کار نیز این است که با کوچکتر شدن ضرایب
# دبلیو
# از وقوع
# اورفیتینگ
# جلوگیری کنیم. زیرا که به ازای حالاتی که در آن 
# دبلیو
# بسیار بزرگ می‌شود آن وقت اورفیتینگ رخ می‌دهد.

# ### 7: The Squared Case:
# first we create the new train and test data:

# In[223]:


X_train_new = np.c_[X_train, X_train**2]
X_test_new = np.c_[X_test, X_test**2]


# In[224]:


x = [i/200 for i in range(21)]
y = []
for lam in x:
    model.fit(X_train_new,y_train,lam)
    w = model.get_param()
    y.append(np.sqrt(w.T @ w))
    
plt.plot(x,y);


# In[225]:


x = [i/200 for i in range(21)]
y = []
for lam in x:
    model.fit(X_train_new,y_train,lam)
    w = model.evaluate(X_test_new, y_test)
    y.append(w)
    
plt.plot(x,y);


# عبارت رگولاریزیشن در عبارت هدف مسئله بهینه سازی باعث می‌شود که به نوعی یک پنالتی به ازای مقادیری که در آن مقادیر درایه‌های
# دبلیو
# بزرگ می‌شود ایجاد می‌کند تا به این ترتیب جواب مسئله بهینه سازی به ازای حالتی رخ بدهد که تا حد ممکن ضرایب
# دبلیو
# کوچکتر بشوند. و هدف ما از انجام این کار نیز این است که با کوچکتر شدن ضرایب
# دبلیو
# از وقوع
# اورفیتینگ
# جلوگیری کنیم. زیرا که به ازای حالاتی که در آن 
# دبلیو
# بسیار بزرگ می‌شود آن وقت اورفیتینگ رخ می‌دهد.

# ### 8: The Cubic Case:
# first we create the new train and test data:

# In[226]:


X_train_new = np.c_[X_train, X_train**2, X_train**3]
X_test_new = np.c_[X_test, X_test**2, X_test**3]


# In[227]:


x = [i/200 for i in range(21)]
y = []
for lam in x:
    model.fit(X_train_new,y_train,lam)
    w = model.get_param()
    y.append(np.sqrt(w.T @ w))
    
plt.plot(x,y);


# In[228]:


x = [i/200 for i in range(21)]
y = []
for lam in x:
    model.fit(X_train_new,y_train,lam)
    w = model.evaluate(X_test_new, y_test)
    y.append(w)
    
plt.plot(x,y);


# عبارت رگولاریزیشن در عبارت هدف مسئله بهینه سازی باعث می‌شود که به نوعی یک پنالتی به ازای مقادیری که در آن مقادیر درایه‌های
# دبلیو
# بزرگ می‌شود ایجاد می‌کند تا به این ترتیب جواب مسئله بهینه سازی به ازای حالتی رخ بدهد که تا حد ممکن ضرایب
# دبلیو
# کوچکتر بشوند. و هدف ما از انجام این کار نیز این است که با کوچکتر شدن ضرایب
# دبلیو
# از وقوع
# اورفیتینگ
# جلوگیری کنیم. زیرا که به ازای حالاتی که در آن 
# دبلیو
# بسیار بزرگ می‌شود آن وقت اورفیتینگ رخ می‌دهد.

# همچنین دلیل این که در بخش 7 و 8، رگولاریزیشن باعث می‌شود که نتیجه مدلسازی چه در کاهش وزن نرم دبلیو اثرگذار باشد و هم خطای پیش بینی را کاهش می‌دهد این است که در این دو بخش، با اضافه کردن چندین متغیر توضیحی جدید، مدلمان در معرض بیشتر خطر اورفیتینگ قرار می‌گیرد و برای همین استفاده از رگولاریزیشن در این جا بسیار موثر می‌شود و باعث بهبود مدل می‌شود. اما در بخش 4 که متغیرهای توان دو و سه را اضافه نکردیم، مشاهده کردیم که خطای پیش بینی اتفاقا با افزایش ضریب لامبدا بیشتر نیز شد و به نوعی در آن جا ضرورت استفاده از رگولاریزشن کمتر است.

# # gradient descent with best learning rate

# Minimize 
# $$
# \frac{1}{2} (Y-\phi W)^T(Y-\phi W) + \frac{1}{2} \lambda W^TW
# $$
# <li> 1-write down gradient descent update formulation </li>
# <li> 2-use hessian matrix to obtain learning rate instead of manually set it. for better underestanding read about newton raphson method</li>
# <li> 3-Plot the target value and the predicted value based on ‘LSTAT’, ‘DIS’, and any other
# features so that you can see how the distributions vary</li>
# <li> 4-plot regularization - weights_norm with lambda between 0 to 10 with step 0.1 </li>
# <li> 5-plot regularization - test_error with lambda between 0 to 10 with step 0.1 </li>
# <li> 6-explain effect of regularization </li>
# <li> 7-add square of each feature to data set and repeat 4,5,6</li>
# <li> 8-add square and power of three of each feature to data set and repeat 4,5,6</li>
# <li> compare part $7^{th}$ test error and previous one <b>explain how 7 and 8 are helping model to do better work </li>
#     

# In[229]:


data,X,y = get_data_normalized()
X_train, X_test, y_train, y_test = split_data(X,y,0.25)


# In[230]:


#this is just template you are free to implement is however you want.
class LinearRegressionModel:#phi is phi = lambda X : np.c_[np.ones(X.shape[0]),X] for adding bias term to data or 
    # any other features to data (this is just suggestion you are free to do whatever you want.)
    def __init__(self):
        self.phi = lambda X : np.c_[np.ones(X.shape[0]),X]
        self.w = []
        pass
    
    def fit(self,X,y,regularization,steps=10):
        """
        get X and y train and learn the parameter by the gradient descent.
        
        """
        self.w=np.zeros(X.shape[1]+1)
        phi=self.phi(X)
        for i in range(steps):
            gradient = (phi.T @ phi + regularization * np.identity(X.shape[1]+1)) @ self.w - phi.T @ y
            hessian = phi.T @ phi + regularization * np.identity(X.shape[1]+1)
            self.w = self.w - np.linalg.inv(hessian) @ gradient
        
    def evaluate(self,X,y):
        """
        get X and y and calculate error.
        """
        r = np.linalg.norm(y-self.transform(X))**2/len(y)
#         r = ((y - self.phi(X) @ self.w).T @ (y - self.phi(X) @ self.w))/X.shape[0]
        return r
    def transform(self,X):
        """
        get X and calculate Phi(X)W as predicted values.
    
        """
        return self.phi(X) @ self.w
    
    def get_param(self):
        "return w "
        return self.w


# ### 3:

# In[231]:


model = LinearRegressionModel()
model.fit(X_train,y_train,0)
yhat = model.transform(X_test)

plt.scatter(X_test.LSTAT, y_test, label='Target Value')
plt.scatter(X_test.LSTAT, yhat, label ='Predictions')
plt.xlabel("LSTAT")
plt.legend();
plt.show()


# In[232]:


plt.scatter(X_test.DIS, y_test, label='Target Value')
plt.scatter(X_test.DIS, yhat, label ='Predictions')
plt.xlabel("DIS")
plt.legend();
plt.show()


# In[233]:


plt.scatter(X_test.AGE, y_test, label='Target Value')
plt.scatter(X_test.AGE, yhat,label ='Predictions')
plt.xlabel("Age")
plt.legend();
plt.show()


# In[234]:


plt.scatter(X_test.CRIM, y_test, label='Target Value')
plt.scatter(X_test.CRIM, yhat,label ='Predictions')
plt.xlabel("CRIM")
plt.legend();
plt.show()


# In[235]:


plt.scatter(X_test.PTRATIO, y_test, label='First')
plt.scatter(X_test.PTRATIO, yhat,label ='Second')
plt.xlabel("PTRATIO")
plt.legend();
plt.show()


# ### 4:

# In[236]:


x = [i/200 for i in range(21)]
y = []
for lam in x:
    model.fit(X_train,y_train,lam)
    w = model.get_param()
    y.append(np.sqrt(w.T @ w))
    
plt.plot(x,y);
plt.xlabel("Lambda")
plt.ylabel("Weights Norm")
plt.show()


# ### 5:

# In[237]:


x = [i/200 for i in range(21)]
y = []
for lam in x:
    model.fit(X_train,y_train,lam)
    w = model.evaluate(X_test, y_test)
    y.append(w)
    
plt.plot(x,y);
plt.xlabel("Lambda")
plt.ylabel("MSE")
plt.show()


# ### 6:
# عبارت رگولاریزیشن در عبارت هدف مسئله بهینه سازی باعث می‌شود که به نوعی یک پنالتی به ازای مقادیری که در آن مقادیر درایه‌های
# دبلیو
# بزرگ می‌شود ایجاد می‌کند تا به این ترتیب جواب مسئله بهینه سازی به ازای حالتی رخ بدهد که تا حد ممکن ضرایب
# دبلیو
# کوچکتر بشوند. و هدف ما از انجام این کار نیز این است که با کوچکتر شدن ضرایب
# دبلیو
# از وقوع
# اورفیتینگ
# جلوگیری کنیم. زیرا که به ازای حالاتی که در آن 
# دبلیو
# بسیار بزرگ می‌شود آن وقت اورفیتینگ رخ می‌دهد.

# ### 7: The Squared Case:
# first we create the new train and test data:

# In[238]:


X_train_new = np.c_[X_train, X_train**2]
X_test_new = np.c_[X_test, X_test**2]


# In[239]:


x = [i/200 for i in range(21)]
y = []
for lam in x:
    model.fit(X_train_new,y_train,lam)
    w = model.get_param()
    y.append(np.sqrt(w.T @ w))
    
plt.plot(x,y);


# In[240]:


x = [i/200 for i in range(21)]
y = []
for lam in x:
    model.fit(X_train_new,y_train,lam)
    w = model.evaluate(X_test_new, y_test)
    y.append(w)
    
plt.plot(x,y);


# عبارت رگولاریزیشن در عبارت هدف مسئله بهینه سازی باعث می‌شود که به نوعی یک پنالتی به ازای مقادیری که در آن مقادیر درایه‌های
# دبلیو
# بزرگ می‌شود ایجاد می‌کند تا به این ترتیب جواب مسئله بهینه سازی به ازای حالتی رخ بدهد که تا حد ممکن ضرایب
# دبلیو
# کوچکتر بشوند. و هدف ما از انجام این کار نیز این است که با کوچکتر شدن ضرایب
# دبلیو
# از وقوع
# اورفیتینگ
# جلوگیری کنیم. زیرا که به ازای حالاتی که در آن 
# دبلیو
# بسیار بزرگ می‌شود آن وقت اورفیتینگ رخ می‌دهد.

# ### 8: The Cubic Case:
# first we create the new train and test data:

# In[241]:


X_train_new = np.c_[X_train, X_train**2, X_train**3]
X_test_new = np.c_[X_test, X_test**2, X_test**3]


# In[242]:


x = [i/200 for i in range(21)]
y = []
for lam in x:
    model.fit(X_train_new,y_train,lam)
    w = model.get_param()
    y.append(np.sqrt(w.T @ w))
    
plt.plot(x,y);


# In[243]:


x = [i/200 for i in range(21)]
y = []
for lam in x:
    model.fit(X_train_new,y_train,lam)
    w = model.evaluate(X_test_new, y_test)
    y.append(w)
    
plt.plot(x,y);


# عبارت رگولاریزیشن در عبارت هدف مسئله بهینه سازی باعث می‌شود که به نوعی یک پنالتی به ازای مقادیری که در آن مقادیر درایه‌های
# دبلیو
# بزرگ می‌شود ایجاد می‌کند تا به این ترتیب جواب مسئله بهینه سازی به ازای حالتی رخ بدهد که تا حد ممکن ضرایب
# دبلیو
# کوچکتر بشوند. و هدف ما از انجام این کار نیز این است که با کوچکتر شدن ضرایب
# دبلیو
# از وقوع
# اورفیتینگ
# جلوگیری کنیم. زیرا که به ازای حالاتی که در آن 
# دبلیو
# بسیار بزرگ می‌شود آن وقت اورفیتینگ رخ می‌دهد.

# همچنین دلیل این که در بخش 7 و 8، رگولاریزیشن باعث می‌شود که نتیجه مدلسازی چه در کاهش وزن نرم دبلیو اثرگذار باشد و هم خطای پیش بینی را کاهش می‌دهد این است که در این دو بخش، با اضافه کردن چندین متغیر توضیحی جدید، مدلمان در معرض بیشتر خطر اورفیتینگ قرار می‌گیرد و برای همین استفاده از رگولاریزیشن در این جا بسیار موثر می‌شود و باعث بهبود مدل می‌شود. اما در بخش 4 که متغیرهای توان دو و سه را اضافه نکردیم، مشاهده کردیم که خطای پیش بینی اتفاقا با افزایش ضریب لامبدا بیشتر نیز شد و به نوعی در آن جا ضرورت استفاده از رگولاریزشن کمتر است.

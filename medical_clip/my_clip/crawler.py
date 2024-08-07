
import requests as rq
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def Fill_database(term,max_res):
    url = f'https://medpix.nlm.nih.gov/search?allen=true&allt=true&alli=true&query={term}'

    driver = webdriver.Chrome()
    driver.get(url)

    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        # Scroll to the bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        # Wait for some time to let new data load
        time.sleep(0.5)
        # Calculate new scroll height and compare with the last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            # Bottom of the page reached
            break
        last_height = new_height

    elements = driver.find_element(By.XPATH, '/html/body/div[6]/div[3]/div[2]/div')
    cow_containers = elements.find_elements(By.CLASS_NAME, 'cow-container')

    data = []

    for i,cow_container in enumerate(cow_containers):    
        try:
            if len(data) + 1 > int(max_res):
                break
            img_element = cow_container.find_element(By.XPATH, './/img[@class="image ng-scope"]')
            image_url = img_element.get_attribute('src')
            wait = WebDriverWait(driver, 100)
            img_element = wait.until(EC.element_to_be_clickable(img_element))
            img_element.click()

            popup = elements.find_element(By.XPATH, '//div[@class="cow-container    selected info"]')
            popup = popup.find_element(By.XPATH, './div/div/div[2]/div[2]/div[1]/div/a')
            popup = wait.until(EC.element_to_be_clickable(popup))
            popup.click()

            driver.switch_to.window(driver.window_handles[-1])
            time.sleep(2)
            text = ''
       
            pairent = driver.find_element(By.XPATH, '/html/body/div[6]/div[3]/div[1]/div[2]/div[2]/div[1]/div')

            #child_divs = pairent.find_elements(By.CSS_SELECTOR, 'div[ng-if="encounter.findings"]')
            child_divs = pairent.find_elements(By.TAG_NAME, 'div')
            for div in child_divs:
                text += ' ' + div.text
            text = text.replace('\n', ' ')
            text = text.replace('\r', ' ')
            text = text.replace('\t', ' ')
        
            data.append([text,image_url])   
            driver.close()
            driver.switch_to.window(driver.window_handles[0])   
        except:
            driver.switch_to.window(driver.window_handles[0])
    driver.quit()
    return data

#writecsv(data,'prototype_db.csv')

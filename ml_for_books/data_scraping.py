# Scraping http://books.toscrape.com/ for data analysis later
# Author: Prof. D (4/19/18) - (4/21/18)

# Importing necessary libraries
import bs4 as bs
import urllib

# Getting all book prices
num_pages = 50
prices = []
for i in range(1, num_pages+1):
    sauce = urllib.request.urlopen('http://books.toscrape.com/catalogue/page-{}.html'.format(i)).read()
    soup = bs.BeautifulSoup(sauce, 'lxml')
    
    divs = soup.find_all('div', class_='product_price')
    for div in divs:
        p_tags = div.find_all('p', class_='price_color')
        for p_tag in p_tags:
            price = list(p_tag.text)
            price.remove(price[0])
            price = float(''.join(price))
            prices.append(price)
            
# Book titles
book_titles = []
for i in range(1, num_pages+1):
    sauce = urllib.request.urlopen('http://books.toscrape.com/catalogue/page-{}.html'.format(i)).read()
    soup = bs.BeautifulSoup(sauce, 'lxml')
    
    articles = soup.find_all('article', class_='product_pod')
    for article in articles:
        imgs = article.find_all('img', class_='thumbnail')
        for img in imgs:
            book_titles.append(img.get('alt'))

# Links for each book. Has unique info on each
import re

book_links = []
individual_book_link = 'http://books.toscrape.com/catalogue/{}/index.html'
num_books=1000
for i in range(1, num_books+1):
    book_title = book_titles[i-1].split()
    book_title = [word.lower() for word in book_title]
    for j,word in enumerate(book_title):
        word_split = (re.split("[^a-zA-Z0-9-]*", word))
        word_split = list(filter(('').__ne__, word_split))
        word = ''.join(word_split)
        book_title[j] = word
    book_title_link_formating = '-'.join(book_title)+'_'+str( (num_books-(i-1)) )
    book_links.append(individual_book_link.format(book_title_link_formating))

#Fixes double dash error in splitting
for i,link in enumerate(book_links):
    if '--' in link:
        link = link.replace('--', '-')
        book_links[i] = link

#Individual fixes (there's only a handful. Mostly issues w/ accent marks)
book_links[135] = 'http://books.toscrape.com/catalogue/the-white-cat-and-the-monk-a-retelling-of-the-poem-pangur-ban_865/index.html'
book_links[175] = 'http://books.toscrape.com/catalogue/poses-for-artists-volume-1-dynamic-and-sitting-poses-an-essential-reference-for-figure-drawing-and-the-human-form_825/index.html'
book_links[296] = 'http://books.toscrape.com/catalogue/i-know-what-im-doing-and-other-lies-i-tell-myself-dispatches-from-a-life-under-construction_704/index.html'
book_links[465] = 'http://books.toscrape.com/catalogue/i-know-what-im-doing-and-other-lies-i-tell-myself-dispatches-from-a-life-under-construction_704/index.html'
book_links[541] = 'http://books.toscrape.com/catalogue/at-the-existentialist-cafe-freedom-being-and-apricot-cocktails-with-jean-paul-sartre-simone-de-beauvoir-albert-camus-martin-heidegger-edmund-husserl-karl-jaspers-maurice-merleau-ponty-and-others_459/index.html'
book_links[806] = 'http://books.toscrape.com/catalogue/midnight-riot-peter-grant-rivers-of-london-books-1_194/index.html'

# Info from each book's page's datatable
rating_dict = {'star-rating One':1, 'star-rating Two':2, 'star-rating Three':3, 'star-rating Four':4, 'star-rating Five':5}
book_dict = {}
start=0
j=0
end = 20
for i in range(1, num_pages+1):
    sauce = urllib.request.urlopen('http://books.toscrape.com/catalogue/page-{}.html'.format(i)).read()
    soup = bs.BeautifulSoup(sauce, 'lxml')
    
    books_on_page_i = book_links[start:end]
    for link in books_on_page_i:
        try:
            sauce_book = urllib.request.urlopen(link).read()
        except:
            print(link)
            pass
        soup_book = bs.BeautifulSoup(sauce_book, 'lxml')
        data_table = soup_book.find('table', class_="table table-striped")
        num_available = data_table.find_all('td')[-2].text
        num_available = list(num_available.split()[2])
        num_available.remove(num_available[0])
        num_available = int(''.join(num_available))
        
        p_tags = soup_book.find_all('p')
        p_classes = [' '.join(p_tag.get('class')) for p_tag in p_tags if p_tag.get('class') != None]
        rating = rating_dict[p_classes[2]]
        
        book_dict[book_titles[j]] = {}
        book_dict[book_titles[j]]['num_available'] = num_available
        book_dict[book_titles[j]]['rating'] = rating
        book_dict[book_titles[j]]['price'] = prices[j]
        
        j+=1
        
    start=end
    end+=20
    
# Writing the data to a txt file
with open('book_data.txt', 'w') as f:
    f.write('num_available,price,rating \n')
    for book in book_titles:
        f.write('{},{},{} \n'.format(book_dict[book]['num_available'],book_dict[book]['price'],book_dict[book]['rating']))

#Might consider adding in page_num as a feature later.

    

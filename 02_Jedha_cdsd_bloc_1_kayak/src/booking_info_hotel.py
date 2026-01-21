import logging
import scrapy
from scrapy.crawler import CrawlerProcess
import json

class BookingDetailsSpider(scrapy.Spider):
    name = "booking_details"
    
    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'ROBOTSTXT_OBEY': False,
        'CONCURRENT_REQUESTS': 1,
        'DOWNLOAD_DELAY': 3,
        'RANDOMIZE_DOWNLOAD_DELAY': True,
        'DOWNLOAD_TIMEOUT': 60,
        'RETRY_TIMES': 3,
    }
    
    def start_requests(self):
        """Charge les URLs depuis le fichier JSON"""
        # Chemin vers votre fichier JSON avec les URLs
        json_path = r'..\data\all_cities_urls_hotels.json'
        
        # Charger le JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            hotels_data = json.load(f)
        
        self.logger.info(f"📂 Chargement de {len(hotels_data)} URLs d'hôtels")
        
        # Créer une requête pour chaque URL
        for hotel in hotels_data:
            url = hotel['url']
            city = hotel['city']
            
            yield scrapy.Request(
                url=url,
                callback=self.parse_hotel,
                meta={'city': city, 'url': url},
                errback=self.handle_error
            )
    
    def parse_hotel(self, response):
        """Extrait les détails d'un hôtel"""
        city = response.meta['city']
        url = response.meta['url']
        
        # Log pour suivi
        self.logger.info(f"🏨 Extraction: {response.url}")
        
        # === NOM DE L'HÔTEL ===
        # Plusieurs sélecteurs possibles
        nom = response.css('h2.pp-header__title::text').get()
        if not nom:
            nom = response.css('h2[data-testid="property-name"]::text').get()
        if not nom:
            nom = response.xpath('//*[@id="hp_hotel_name"]/div/h2/text()').get()
            # nom = response.xpath('//h2[@class="hp__hotel-name"]/text()').get()
        if not nom:
            nom = response.css('h1.d2fee87262::text').get()
        
        # === Note ===
        # Note globale (ex: 8.5)
        note = response.css('div.b5cd09854e::text').get()
        if not note:
            note = response.css('div[data-testid="review-score-component"] div::text').get() 
        if not note:
            note = response.xpath('//*[@id="js--hp-gallery-scorecard"]/a/div/div/div/div[2]/text()').get()
        
            # note = response.xpath('//div[@class="b5cd09854e d10a6220b4"]/text()').get() #//*[@id="js--hp-gallery-scorecard"] //*[@id="js--hp-gallery-scorecard"]/a/div/div/div/div[1]  //*[@id="js--hp-gallery-scorecard"]/a/div/div/div/div[2]
        
        # # Nombre d'avis
        # nb_avis = response.css('div.d8eab2cf7f::text').get()
        # if not nb_avis:
        #     nb_avis = response.css('div[data-testid="review-score-component"] span::text').get()
        
        # === ADRESSE COMPLÈTE ===
        adresse = response.css('span.hp_address_subtitle::text').get()
        if not adresse:
            adresse = response.css('span[data-node_tt_id="location_score_tooltip"]::text').get()
        if not adresse:
            adresse = response.xpath('//*[@id="wrap-hotelpage-top"]/div[3]/div/div/div/div/div/span[1]/button/div/text()').get()
            # adresse = response.xpath('//span[@data-node_tt_id="location_score_tooltip"]/text()').get()
        if not adresse:
            # Essayer de construire l'adresse depuis plusieurs éléments
            adresse_parts = response.css('p.address span::text').getall()
            adresse = ' '.join(adresse_parts) if adresse_parts else None
        
        # === DESCRIPTION ===
        # La description est souvent dans plusieurs paragraphes
        description_parts = response.css('div#property_description_content p::text').getall()
        if not description_parts:
            description_parts = response.css('div.hp_desc_main_content p::text').getall()
        if not description_parts:
            description_parts = response.xpath('//*[@id="basiclayout"]/div/div[3]/div[1]/div[1]/div[1]/div[1]/div/div/p[1]/text()').getall()
            # description_parts = response.xpath('//div[@id="property_description_content"]//p/text()').getall()
        
        description = ' '.join(description_parts).strip() if description_parts else None
        
        # Si pas de description complète, essayer un texte plus court
        if not description:
            description = response.css('div.a53cbfa6de::text').get()
        
        # # === INFORMATIONS SUPPLÉMENTAIRES (BONUS) ===
        # # Prix (si disponible)
        # prix = response.css('span.prco-valign-middle-helper::text').get()
        # if not prix:
        #     prix = response.css('div.bui-price-display__value::text').get()
        
        # # Type d'hébergement
        # type_hebergement = response.css('span.hp__hotel-type-badge::text').get()
        
        # # Équipements populaires
        # equipements = response.css('div.important_facility span::text').getall()
        # if not equipements:
        #     equipements = response.css('div.hp-description ul li::text').getall()
        
        # === CONSTRUCTION DU RÉSULTAT ===
        hotel_data = {
            'ville': city,
            'url': url,
            'nom': nom.strip() if nom else 'Non disponible',
            'note': note.strip() if note else 'Non disponible',
            # 'nombre_avis': nb_avis.strip() if nb_avis else 'Non disponible',
            'adresse': adresse.strip() if adresse else 'Non disponible',
            'description': description if description else 'Non disponible'#,
            # 'prix': prix.strip() if prix else 'Non disponible',
            # 'type_hebergement': type_hebergement.strip() if type_hebergement else 'Non disponible',
            # 'equipements': equipements[:5] if equipements else []  # Top 5 équipements
        }
        
        # Log de confirmation
        self.logger.info(f"✅ Extrait: {hotel_data['nom']} - Note: {hotel_data['note']}")
        
        yield hotel_data
    
    def handle_error(self, failure):
        """Gestion des erreurs"""
        self.logger.error(f"❌ Erreur lors du scraping: {failure.value}")
        self.logger.error(f"URL concernée: {failure.request.url}")


# === CONFIGURATION ET LANCEMENT ===
if __name__ == '__main__':
    # Fichier de sortie pour les détails
    output_path = r'C:\Users\alber\Desktop\visual_studio_code\dossier_jedha\Jedha_Full_stack\03_Data_Collection_&_Management_(DCM)\00- projet kayak\data\hotels_details.json'
    
    # Configuration du processus
    process = CrawlerProcess(settings={
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'LOG_LEVEL': logging.INFO,
        'FEEDS': {
            output_path: {
                "format": "json",
                'overwrite': True,
                'encoding': 'utf-8',
                'indent': 2
            },
        }
    })
    
    # Lancement du spider
    process.crawl(BookingDetailsSpider)
    process.start()
import logging
import os
import csv
from wordfreq import top_n_list
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from better_profanity import profanity
import textwrap
from PIL import Image, ImageDraw, ImageFont


class MultilingualList:

    # Paths
    root_path = os.path.dirname( os.path.realpath( __file__ ) )
    res_dir_name = 'resources'
    res_dir_name = os.path.join( root_path, res_dir_name )
    fr_genders_name = 'french-noun-gender.csv'
    fr_genders_path = os.path.join( res_dir_name, fr_genders_name )
    sp_genders_name = 'es_ancora-ud-dev.conllu.txt'
    sp_genders_path = os.path.join( res_dir_name, sp_genders_name )
    csv_dir_name = 'csv'
    csv_dir_path = os.path.join( root_path, csv_dir_name )
    synsets_name = 'synsets.csv'
    synsets_path = os.path.join( csv_dir_path, synsets_name )
    png_dir_name = 'png'
    png_dir_path = os.path.join( root_path, png_dir_name )

    # Variables
    nltk_resources = [ [ 'corpora', 'wordnet' ],
                       [ 'corpora', 'omw-2.0' ],
                       [ 'corpora', 'stopwords' ],
                       [ 'taggers', 'averaged_perceptron_tagger_eng' ] ]
    en_n = 3913
    synset_dict = {}
    genders = { 'fr': None, 'sp': None }
    fr_art = { 'masculine': 'le', 'feminine': 'la', 'epicene': 'le/la'}
    sp_art = { 'NCF0000': 'la',
                 'NCFN000': 'la',
                 'NCFS000': 'la',
                 'NCFP000': 'las',
                 'NC00000': 'el',
                 'NCCN000': 'el',
                 'NCMN000': 'el',
                 'NCMS000': 'el',
                 'NCMP000': 'los',
                 'NCCS000': 'el/la',
                 'NCCP000': 'los/las' }
    n_slices = 3
    allowed_pos = [ 'n', 'a', 'v', 'r' ]
    profanity_allowed = False
    csv_header = [ 'synset-id', 'pos', 'en', 'fr', 'sp' ]
    synset_rows = []
    width = 600
    height = 1000
    font_size = 50
    text_margin = 40
    text_wrap = 17
    text_y = 200
    bg_color = 0
    text_color = 255
    font_family = [ 'cour.ttf',
                    'Courier New.ttf',
                    'Consolas.ttf',
                    'DejaVuSansMono.ttf',
                    'LiberationMono-Regular.ttf']
    font = None
    for font_family in font_family:
        try:
            font = ImageFont.truetype( font_family, size=font_size )
            break
        except IOError:
            continue
        if font is None:
            font = ImageFont.load_default()


    def __init__( self, en_n=None, n_slices=None, profanity_allowed=None ):

        if en_n is not None:
            self.en_n = en_n

        if n_slices is not None:
            self.n_slices = n_slices

        if profanity_allowed is not None:
            self.profanity_allowed = profanity_allowed

        # Logger
        self.logger = logging.getLogger( __name__ )
        self.logger.setLevel( logging.DEBUG )
        log_format = '%(levelname)s: [ %(funcName)s ] %(message)s'
        logging.basicConfig( format=log_format )


    def synset_csv( self ):
        # Make a synset csv

        self.nltk_check()
        self.en_stopwords = stopwords.words( 'english' )

        self.en_words()
        self.fr_genders()
        self.sp_genders()
        self.find_others()
        self.make_rows()
        self.write_csv()
        self.report()


    def nltk_check( self ):
        # Check for NLTK resources

        self.logger.info( 'Check for NLTK resources' )

        for nltk_res in self.nltk_resources:
            self.logger.info( f'nltk_res {nltk_res}' )
            try:
                path = nltk.data.find( f'{'/'.join(nltk_res)}.zip' )
                self.logger.info( f'path {path}' )
            except LookupError:
                self.logger.info( f'{'/'.join(nltk_res)} not found' )
                nltk.download( nltk_res[1] )

    def en_words( self ):
        # English words as self.en_words

        self.logger.info( 'English words' )
        
        en_words = top_n_list( 'en', self.en_n )
        en_words = [ w for w in en_words if w.lower() not in self.en_stopwords ]
        
        self.en_words = en_words


    def fr_genders( self ):
        # French genders as self.genders['fr']

        self.logger.info( 'French genders' )
        
        with open( self.fr_genders_path,
                   mode='r',
                   newline='',
                   encoding='utf-8' ) as csv_f:

            csv_reader = csv.reader( csv_f )
            fr_genders_list = list( csv_reader )

            fr_genders_list = fr_genders_list[1:] # omit header

            for i, row in enumerate(fr_genders_list):
                fr_genders_list[i][1] = self.fr_art[ row[1] ]
            fr_genders = { row[0]: row[1] for row in fr_genders_list }

        self.genders['fr'] = fr_genders
    
    
    def sp_genders( self ):
        # Spanish genders as self.genders['sp']

        self.logger.info( 'Spanish genders' )

        with open( self.sp_genders_path,
                   mode='r',
                   newline='',
                   encoding='utf-8' ) as tsv_f:
            
            tsv_reader = csv.reader( tsv_f, delimiter='\t' )
            sp_genders_list = list( tsv_reader )
            
            # Omit rows without a EAGLES part-of-speech tag
            sp_genders_list = [ r for r in sp_genders_list if len(r) >= 4 ]
            
            sp_genders_list = [
                                [ r[1], r[4] ] for r in sp_genders_list \
                                if r[4].upper() in self.sp_art.keys() ]

            for i, row in enumerate( sp_genders_list ):
                sp_genders_list[i][1] = self.sp_art[ row[1].upper() ]
            sp_genders = { row[0]: row[1] for row in sp_genders_list }

        self.genders['sp'] = sp_genders


    def gender( self, word, fr_sp ):
        # Return the same word if there isn't an entry for the gender
        # else return the word prepended with the gendered article

        return word if word not in self.genders[fr_sp] \
            else self.genders[fr_sp][word] + ' ' + word


    def find_others( self ):
        # Find English French and Spanish words based on the synsets the English
        # words have

        en_list = []
        fr_list = []
        sp_list = []

        for word in self.en_words:
            
            self.logger.info( f'word {word}' )
            
            # English French and Spanish initial synset
            synsets = wn.synsets( word )
            
            if len(synsets) == 0:
                continue
            
            synset = synsets[0]
            
            # Synset ID
            synset_id = f'{str(synset.offset()).zfill(8)}-{synset.pos()}'
                
            self.logger.info( f'synset_id {synset_id}' )
            
            en = synset.lemma_names( lang='eng' )
            fr = synset.lemma_names( lang='fra' )
            sp = synset.lemma_names( lang='spa' )

            en = self.sanitize_words( en )
            fr = self.sanitize_words( fr )
            sp = self.sanitize_words( sp )

            pos = synset.pos()

            # No duplicates. No words not in Englsh French and Spanish. No
            # proper nouns. No parts of speech other than noun, adjective, verb
            # and adverb. No digits. If no profanity allowed then no profanity.
            if en in en_list or fr in fr_list or sp in sp_list \
                or len(en) == 0 or len(fr) == 0 or len(sp) == 0 \
                or any( [ self.has_proper(x) for x in [ en, fr, sp ] ] ) \
                or pos not in self.allowed_pos \
                or any( [ self.has_numeric(x) for x in [ en, fr, sp ] ]  ) \
                or ( not self.profanity_allowed \
                     and profanity.contains_profanity(' '.join(en)) ):
                
                continue

            en_list.append(en)
            fr_list.append(fr)
            sp_list.append(sp)

            self.synset_dict[synset_id] = {}
            
            self.synset_dict[synset_id]['pos'] = pos
            self.synset_dict[synset_id]['en'] = en[0] if len(en) == 1 \
                                            else ', '.join(en)
            
            if pos == 'n':
                fr = [ self.gender(w, 'fr') for w in fr ]
                sp = [ self.gender(w, 'sp') for w in sp ]
            self.synset_dict[synset_id]['fr'] = fr[0] if len(fr) == 1 \
                                            else ', '.join(fr)
            self.synset_dict[synset_id]['sp'] = sp[0] if len(sp) == 1 \
                                            else ', '.join(sp)

            self.logger.info( f'self.synset_dict[synset_id] {self.synset_dict[synset_id]}' )

        self.synsets_n = len( self.synset_dict )


    def sanitize_words( self, word_list ) :
        # Limit list to n_slices and insert underscores for spaces

        word_list = word_list[ 0:self.n_slices ]
        return [ w.replace('_', ' ') for w in word_list ]
    
    def has_proper( self, word_list ):
        # Return True if there are proper nouns in the words

        word_list = [ w for w in word_list if not w.isnumeric()]
        words = ' '.join( word_list )
        self.logger.info( f'words {words}')
        if any( char.isupper() for char in words ):
            return True

        tags = pos_tag( word_tokenize( words ) )

        return any( tag in ( 'NNP', 'NNPS' ) for word, tag in tags )
    
    def has_numeric( self, list_ ):
        # Return True if there is a numeric element

        return any( x.isnumeric() for x in list_ )
    
    def make_rows( self ):
        # Synsets rows

        self.logger.info( 'Synsets rows' )
        
        self.logger.info( f'len(self.synset_dict) {len(self.synset_dict)}' )

        self.synset_rows = [ self.csv_header ]
        for synset_id, w in self.synset_dict.items():
            self.synset_rows.append( [ synset_id, w['pos'], w['en'], w['fr'], w['sp'] ] )

        self.logger.info( f'len(self.synset_rows) {len(self.synset_rows)}' )


    def write_csv( self ):
        # Write self.synset_rows to a csv
        with open( self.synsets_path, mode='w', newline='', encoding='utf-8' ) \
                as csv_f:
            writer = csv.writer( csv_f )

            for row in self.synset_rows:
                writer.writerow( row )

    def report( self ):
        # Salient numbers

        self.logger.info( 'Salient numbers' )
        self.logger.info( f'self.profanity_allowed {self.profanity_allowed}' )
        self.logger.info( f'self.en_n {self.en_n}' )
        self.logger.info( f'self.en_stopwords {len(self.en_stopwords)}' )
        self.logger.info( f'self.n_slices {self.n_slices}' )
        self.logger.info( f'self.synsets_n {self.synsets_n}' )

    def citations( self ):
        # Citations

        print( wn.citation() )
        print( wn.citation( 'eng' ) )
        print( wn.citation( 'fra' ) )
        print( wn.citation( 'spa' ) )


    def flashcards( self ):
        # Make flashcards
        
        self.logger.info( 'Make flashcards' )
        
        with open( self.synsets_path,
                   mode='r',
                   newline='',
                   encoding='utf-8' ) as csv_f:
            
            csv_reader = csv.reader( csv_f )
            synsets_list = list( csv_reader )

            synsets_list = synsets_list[1:]
            synsets = { synset[0]: { 'pos': synset[1],
                                     'en': synset[2],
                                     'fr': synset[3],
                                     'es': synset[4] } for synset \
                                                       in synsets_list }

        max_width = self.width - ( 2 * self.text_margin )

        for synset_id, synset in synsets.items():
            # Make a black background image
            # 'L' mode is 8-bit pixels black and white
            image = Image.new( 'L', ( self.width, self.height ), color=self.bg_color )
            draw = ImageDraw.Draw( image )

            text = self.tw( synset['en'] ) + '\n\n' + \
                self.tw( synset['fr'] ) + '\n\n' + \
                self.tw( synset['es'] )
            
            draw.multiline_text( ( self.text_margin, self.text_y ),
                                 text,
                                 fill=self.text_color,
                                 font=self.font )

            image.save( os.path.join( self.png_dir_path, synset_id + '.png' ) )


    def tw( self, string ):
        # Return text-wrapped

        self.logger.info( 'Return text-wrapped' )

        self.logger.info( f'string {string}' )

        return textwrap.fill( string, width=self.text_wrap )


if __name__ == '__main__':
    ML = MultilingualList()
    # ML.synset_csv()
    # ML.flashcards()
    ML.citations()

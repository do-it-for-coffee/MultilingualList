import logging
import os
from pathlib import Path
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
    res_dir_name = 'resources'
    ext_dir_name = 'external'
    int_dir_name = 'internal'
    csv_dir_name = 'csv'
    png_dir_name = 'png'
    omit_name = 'omit.csv'
    g_dep_name = 'gender-dependent.csv'
    fr_genders_name = 'french-noun-gender.csv'
    sp_genders_name = 'es_ancora-ud-dev.conllu.txt'
    synsets_name = 'synsets.csv'
    cit_dir_name = 'citations'
    cit_name = '-citations.bib'
    root_path = os.path.dirname( os.path.realpath( __file__ ) )
    res_dir_path = os.path.join( root_path, res_dir_name )
    ext_dir_path = os.path.join( res_dir_path, ext_dir_name )
    int_dir_path = os.path.join( res_dir_path, int_dir_name )
    csv_dir_path = os.path.join( root_path, csv_dir_name )
    png_dir_path = os.path.join( root_path, png_dir_name )
    cit_dir_path = os.path.join( root_path, cit_dir_name )
    fr_genders_path = os.path.join( ext_dir_path, fr_genders_name )
    sp_genders_path = os.path.join( ext_dir_path, sp_genders_name )
    g_dep_path = os.path.join( int_dir_path, g_dep_name )
    omit_path = os.path.join( int_dir_path, omit_name )
    synsets_path = os.path.join( csv_dir_path, synsets_name )

    # Variables
    nltk_resources = [ [ 'corpora', 'wordnet' ],
                       [ 'corpora', 'omw-2.0' ],
                       [ 'corpora', 'stopwords' ],
                       [ 'taggers', 'averaged_perceptron_tagger_eng' ] ]
    en_n = 3702
    synset_dict = {}
    genders = { 'fr': None, 'sp': None }
    art = { 'fr': { 'masculine': 'le',
                    'feminine': 'la',
                    'epicene': 'le/la' },
            'sp': { 'masculine': 'el',
                    'feminine': 'la',
                    'epicene': 'el/la' } }
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
    sp_g_s = { 'F': 'NCFS000', 'M': 'NCMS000' }
    sp_end = { 'a': 'la',
               'ción': 'la',
               'sión': 'la',
               'dad': 'la',
               'tad': 'la',
               'tud': 'la',
               'ez': 'la',
               'umbre': 'la',
               'o': 'el',
               'or': 'el',
               'aje': 'el',
               'ma': 'el',
               'l': 'el',
               'r': 'el',
               's': 'el',
               'n': 'el',
               'd': 'el' }
    n_slices = 2
    allowed_pos = [ 'a', 'v', 'r', 'n' ]
    pos_names = { 'a': 'adj.',
                  'v': 'v.',
                  'r': 'adv.',
                  'n': 'n.' }
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

        Path( self.ext_dir_path ).mkdir(parents=True, exist_ok=True)
        Path( self.int_dir_path ).mkdir(parents=True, exist_ok=True)
        Path( self.csv_dir_path ).mkdir(parents=True, exist_ok=True)
        Path( self.png_dir_path ).mkdir(parents=True, exist_ok=True)
        Path( self.cit_dir_path ).mkdir(parents=True, exist_ok=True)

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

        # Omit Wordnet errors
        with open( self.omit_path,
                    mode='r',
                    newline='',
                    encoding='utf-8' ) as csv_f:
            csv_reader = csv.reader( csv_f )
            self.omit = list( csv_reader )

        with open( self.g_dep_path,
                    mode='r',
                    newline='',
                    encoding='utf-8' ) as csv_f:
            csv_reader = csv.reader( csv_f )
            self.g_dep = list( csv_reader )

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
        
        en_words = top_n_list( 'en', self.en_n, wordlist='best' )
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
            fr_genders_list[i][1] = self.art['fr'][ row[1] ]
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
            sp_genders = list( tsv_reader )
            
        # Omit rows without a EAGLES part-of-speech tag
        sp_genders = [ r for r in sp_genders if len(r) >= 4 ]
        
        sp_genders = [
                            [ r[1], r[4], r[2] ] for r in sp_genders \
                            if r[4].upper() in self.sp_art.keys() ]

        self.logger.info( f'len(sp_genders) {len(sp_genders)}')
        # For words with no entry yet, find if the word is in
        # sp_genders_[2], and have the gender singular form derived
        # since that probably lists that as the plural form.
        to_add = []
        initial = [ row[0] for row in sp_genders ]
        for row in sp_genders:
            # not already listed and
            # to_add is empty or not already in to_add
            if row[2] not in initial \
                    and ( len(to_add) == 0 \
                            or row[2] not in [ row_[0] for row_ in to_add ] ):
                for g, art in self.sp_g_s.items():
                    if g in row[1].upper():
                        to_add.append( [ row[2], art ] )
        sp_genders = [ [ row[0], row[1] ] for row in sp_genders ]
        
        sp_genders = sp_genders + to_add
        self.logger.info( f'len(to_add) {len(to_add)}')
        self.logger.info( f'len(sp_genders) {len(sp_genders)}')

        for i, row in enumerate( sp_genders ):
            sp_genders[i][1] = self.sp_art[ row[1].upper() ]

        sp_genders = { row[0]: row[1] for row in sp_genders }

        self.genders['sp'] = sp_genders


    def gender( self, word, lan, en=None ):
        # Return the same word if there isn't an entry for the gender
        # else return the word prepended with the gendered article

        # Words with different definitions though the same spelling
        # If lan is row[0] and that word is row[1]
        #   if the English is row[2] then row[3]
        for row in self.g_dep:
            if row[0] == lan and row[1] == word:
                if row[2] in en:
                    return self.art[lan][ row[3] ] + ' ' + word

        if word not in self.genders[lan]:
            # Spanish words with no entry find gender from the ending the word
            # has
            if lan == 'sp':
                for end, art in self.sp_end.items():
                    if word.endswith( end ):
                        return art + ' ' + word

            return word

        return self.genders[lan][word] + ' ' + word


    def find_others( self ):
        # Find English French and Spanish words based on the synsets the English
        # words have

        unsorted_synsets = {}
        for word in self.en_words:
            
            self.logger.info( f'unsorted_synsets word {word}' )
            
            # English French and Spanish initial synset
            synsets = wn.synsets( word )
            
            if len(synsets) == 0:
                continue
            
            synset = synsets[0]
            
            # Synset ID
            synset_id = f'{str(synset.offset()).zfill(8)}-{synset.pos()}'
                
            en = synset.lemma_names( lang='eng' )
            fr = synset.lemma_names( lang='fra' )
            sp = synset.lemma_names( lang='spa' )

            en = self.sanitize_words( synset_id, 'en', en  )
            fr = self.sanitize_words( synset_id, 'fr', fr )
            sp = self.sanitize_words( synset_id, 'sp', sp )

            pos = synset.pos()

            unsorted_synsets[synset_id] = { 'pos': pos,
                                            'en': en,
                                            'fr': fr,
                                            'sp': sp }

        # Sort synsets to prioritize adjectives over nominalized adjectives when
        # duplicates are skipped. Float inf places other parts of speech at the
        # end.
        pos_map = { pos: index for index, pos in enumerate(self.allowed_pos) }
        sorted_synsets = dict(
            sorted(
                unsorted_synsets.items(),
                key=lambda item: pos_map.get( item[1]['pos'], float('inf') )
        ) )
        
        for synset_id, synset in sorted_synsets.items():
            self.logger.info( f'sorted_synsets synset_id {synset_id}' )
            self.logger.info( f'sorted_synsets synset {synset}' )
            
            pos = synset['pos']
            en = synset['en']
            fr = synset['fr']
            sp = synset['sp']

            en_list = [ item[1]['en'] for item in self.synset_dict.items() ]
            fr_list = [ item[1]['fr'] for item in self.synset_dict.items() ]
            sp_list = [ item[1]['sp'] for item in self.synset_dict.items() ]

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

            self.synset_dict[synset_id] = {}
            
            self.synset_dict[synset_id]['pos'] = pos
            
            en = en[0] if len(en) == 1 else ', '.join(en)
            self.synset_dict[synset_id]['en'] = en
            
            if pos == 'n':
                fr = [ self.gender(w, 'fr', en=en) for w in fr ]
                sp = [ self.gender(w, 'sp', en=en) for w in sp ]
            
            fr = fr[0] if len(fr) == 1 else ', '.join(fr)
            sp = sp[0] if len(sp) == 1 else ', '.join(sp)

            self.synset_dict[synset_id]['fr'] = fr
            self.synset_dict[synset_id]['sp'] = sp

            self.logger.info( f'self.synset_dict[synset_id] {self.synset_dict[synset_id]}' )

        self.synsets_n = len( self.synset_dict )


    def sanitize_words( self, synset_id, lan, word_list ) :
        # Sanitize provided words

        # Omit based on synset ID no other information is provided and omit
        # based on the synset ID and word if that is provided.
        for word in word_list:
            for o in self.omit:
                if ( len(o) == 1 or ( o[1] == '' or o[2] == '' ) ) \
                        and synset_id == o[0]:
                    word_list = []
                elif synset_id == o[0] \
                        and lan == o[1] \
                        and word == o[2]:
                    word_list = [ w for w in word_list if w != o[2] ]
        
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

        self.logger.info( 'Citations' )

        self.write_bib( 'eng' )
        self.write_bib( 'fra' )


    def write_bib ( self, lan=None ):
        # Write bib
        
        self.logger.info( 'Write bib' )
        self.logger.info( f'lan {lan}' )

        bib = wn.citation( lan )
        
        if lan == 'eng':
            bib_name = 'en' + self.cit_name
        elif lan == 'fra' or lan == 'spa':
            bib_name = 'fr-sp' + self.cit_name


        bib_path = os.path.join( self.cit_dir_path, bib_name )
        with open( bib_path, mode='w', encoding='utf-8' ) as bib_f:
            bib_f.write( bib )

    
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

            text = self.pos_names[ synset['pos'] ] + '\n\n' + \
                self.tw( synset['en'] ) + '\n\n' + \
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
    # ML.citations()
    # ML.synset_csv()
    # ML.flashcards()

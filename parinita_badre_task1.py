from pyspark import SparkContext
import sys


def get_char_matrix(data):
    cmatrix = dict()
    for row in data:
            businessID = businessdict.get(row[1])
            userID = userdict.get(row[0])

            if businessID not in cmatrix:
                cmatrix[businessID] = [userID]
            else:
                cmatrix[businessID].append(userID)
    return cmatrix


def get_businessindex_dict(business):
    bdict = dict()
    c = 0
    for i in business:
        bdict[i] = c
        indexbusinessdict[c] = i
        c += 1
    return bdict


def get_userindex_dict(user):
    udict = dict()
    c = 0
    for i in user:
        udict[i] = c
        c += 1
    return udict


def calculate_hash(x, val):
    a = val[0]
    b = val[1]
    p = val[2]
    min_hash = 999999
    for user_index in x[1]:
        k = (((a * user_index) + b) % p) % totalusers
        if k < min_hash:
            min_hash = k
    return min_hash


def create_bands(x):
    total_bands = list()
    for i in range(bands):
        val = x[1][i * row:(i + 1) * row]
        total_bands.append(((i, tuple(val)), [x[0]]))
    return total_bands


def generate_pairs(signature):
    pairs = []
    length = len(signature[1])
    list_row = list(signature[1])
    list_row.sort()
    for i in range(length):
        for j in range(i + 1, length):
            pairs.append(((list_row[i], list_row[j]), 1))
    return pairs


def jaccard_similarity(x):
    business1 = set(charmatrix[x[0]])
    business2 = set(charmatrix[x[1]])
    intersection = business1 & business2
    union = business1 | business2
    jsimilarity = len(intersection) / len(union)
    return x[0], x[1], jsimilarity


if __name__ == '__main__':


    # Accept input
    input_file_path = sys.argv[1]
    similarity_method = sys.argv[2]
    output_file_path = sys.argv[3]

    sc = SparkContext("local[*]", "Jaccard Similarity")

    if similarity_method == "jaccard":

        # Convert input file to text and remove header
        rdd = sc.textFile(input_file_path)
        header = rdd.first()
        rdd = rdd.filter(lambda x: x != header).map(lambda x: x.split(','))

        # Map all data into rdd and create dictionary of indices
        user = rdd.map(lambda x: x[0]).distinct().collect()
        user.sort()
        business = rdd.map(lambda x: x[1]).distinct().collect()
        business.sort()
        datalist = rdd.collect()

        userdict = get_userindex_dict(user)
        indexbusinessdict = dict()
        businessdict = get_businessindex_dict(business)
        charmatrix = get_char_matrix(datalist)
        totalusers = len(userdict)

        abp_value = [[421, 167, 1610612741], [421, 397, 1610612741], [421, 257, 3145739], [479, 193, 201326611],
                        [659, 193, 100663319], [421, 193, 201326611], [619, 257, 100663319], [619, 139, 1610612741],
                        [389, 397, 100663319], [659, 193, 402653189], [479, 167, 1610612741], [479, 211, 201326611],
                        [983, 139, 201326611], [983, 257, 3145739], [443, 137, 402653189], [929, 397, 3145739],
                        [167, 431, 402653189], [421, 139, 12582917], [761, 131, 3145739], [761, 389, 402653189],
                        [317, 193, 1572869], [241, 139, 393241], [467, 167, 805306457], [109, 167, 786433],
                        [547, 397, 25165843], [109, 191, 12582917], [641, 397, 1610612741], [983, 389, 196613],
                        [641, 373, 402653189], [127, 233, 201326611], [641, 373, 25165843], [389, 107, 786433],
                        [257, 107, 1572869], [491, 149, 201326611], [389, 163, 805306457], [109, 193, 1572869],
                        [967, 277, 25165843], [953, 151, 805306457], [547, 139, 196613], [919, 419, 100663319],
                        [937, 431, 201326611], [661, 383, 1610612741], [109, 107, 6291469], [491, 293, 805306457],
                        [167, 389, 50331653], [757, 397, 50331653], [821, 173, 1610612741], [467, 419, 805306457],
                        [983, 397, 402653189], [317, 193, 393241], [967, 137, 25165843], [677, 173, 196613],
                        [607, 307, 1610612741], [619, 431, 1610612741], [661, 167, 786433], [821, 137, 6291469],
                        [701, 347, 402653189], [919, 137, 393241], [809, 223, 1610612741], [769, 151, 25165843],
                        [467, 439, 1610612741], [509, 383, 786433], [487, 223, 50331653], [409, 211, 98317],
                        [761, 137, 98317], [929, 271, 1572869], [769, 293, 201326611], [157, 139, 1610612741],
                        [443, 331, 1610612741], [769, 149, 3145739], [151, 241, 12582917], [359, 373, 201326611],
                        [487, 223, 100663319], [317, 227, 49157], [509, 331, 1610612741], [479, 271, 786433],
                        [557, 139, 805306457], [761, 347, 805306457], [641, 443, 98317], [821, 307, 393241],
                        [109, 163, 98317], [509, 251, 49157], [811, 233, 805306457], [701, 293, 50331653],
                        [641, 373, 786433], [479, 127, 1572869], [727, 131, 98317], [541, 383, 25165843],
                        [503, 103, 196613], [601, 443, 786433], [239, 233, 100663319], [827, 353, 6291469],
                        [277, 293, 805306457], [487, 173, 1610612741], [191, 223, 3145739], [613, 211, 1572869],
                        [257, 257, 786433], [541, 331, 201326611], [997, 181, 3145739], [613, 439, 6291469],
                        [983, 421, 201326611], [929, 179, 98317], [503, 109, 786433], [967, 107, 100663319],
                        [607, 167, 100663319], [331, 173, 50331653], [223, 197, 6291469], [907, 311, 805306457],
                        [127, 397, 12582917], [557, 179, 100663319], [601, 241, 393241], [953, 131, 196613],
                        [743, 331, 25165843], [601, 211, 49157], [457, 421, 805306457], [151, 197, 1572869],
                        [757, 107, 201326611], [421, 197, 196613], [701, 197, 805306457], [433, 113, 49157]]

        charmat = sc.parallelize(list(charmatrix.items()))
        charmat = charmat.sortBy(lambda x: x[0])
        matrix = charmat.collect()

        signature_matrix = charmat.map(lambda x: (x[0], [calculate_hash(x, val) for val in abp_value]))

        bands = 40
        row = int(len(abp_value) / bands)

        get_pair = signature_matrix.flatMap(create_bands).reduceByKey(lambda x, y: x + y).filter(
            lambda x: len(x[1]) > 1).flatMap(generate_pairs).reduceByKey(lambda x, y: x).map(lambda x: x[0])

        final = get_pair.map(jaccard_similarity).filter(lambda x: x[2] >= 0.5).sortBy(
            lambda x: x[1]).sortBy(lambda x: x[0])

        with open(output_file_path, 'w') as fout:
            fout.write("business_id_1, business_id_2, similarity\n")
            for row in final.collect():
                business1_print = str(indexbusinessdict.get(row[0]))
                business2_print = str(indexbusinessdict.get(row[1]))
                similarity = str(row[2])
                fout.write(business1_print + "," + business2_print + "," + similarity + "\n")
        fout.close()

    else:
        exit(0)

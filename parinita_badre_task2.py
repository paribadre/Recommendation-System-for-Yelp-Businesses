from pyspark.mllib.recommendation import ALS, Rating
import random
import sys
from pyspark import SparkContext
import math

if __name__ == '__main__':

    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    case_id = sys.argv[3]
    output_file_path = sys.argv[4]

    # Model Based Collaborative Filtering
    if int(case_id) == 1:
        sc = SparkContext("local[*]", "Model Based Recommendation system")

        # Store training and testing file in rdd without header
        training = sc.textFile(train_file_path)
        header = training.first()
        training_rdd = training.filter(lambda x: x != header).map(lambda x: x.split(','))

        test_rdd = sc.textFile(test_file_path)
        header_test = test_rdd.first()
        test_rdd = test_rdd.filter(lambda x: x != header_test).map(lambda x: x.split(','))

        # Combine users in train and test data and assign index, store in dict
        user_train = training_rdd.map(lambda x: x[0]).distinct()
        user_test = test_rdd.map(lambda x: x[0]).distinct()
        user = user_train.union(user_test).collect()
        user.sort()

        userdict = dict()
        indexuserdict = dict()
        c = 1
        for i in user:
            userdict[i] = c
            indexuserdict[c] = i
            c += 1

        # Combine business in train and test data and assign index, store in dict
        business_train = training_rdd.map(lambda x: x[1]).distinct()
        business_test = test_rdd.map(lambda x: x[1]).distinct()
        business = business_train.union(business_test).collect()
        business.sort()

        businessdict = dict()
        indexbusinessdict = dict()
        c = 1
        for i in business:
            businessdict[i] = c
            indexbusinessdict[c] = i
            c += 1

        # Convert training rdd to indexes
        training_rdd = training_rdd.map(lambda x: (userdict.get(x[0]), businessdict.get(x[1]), float(x[2])))
        training_ratings = training_rdd.map(lambda x: Rating(x[0], x[1], x[2]))

        # Convert test rdd to indexes
        testdata = test_rdd.map(lambda x: (userdict.get(x[0]), businessdict.get(x[1])))

        # Create model and make predictions
        random.seed(200)
        model = ALS.train(training_ratings, rank=5, iterations=15, lambda_=0.3)
        predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))

        # Create ground truth rdd with indexes and calculate mean
        ground_truth_index = test_rdd.map(lambda x: ((userdict.get(x[0]), businessdict.get(x[1])), float(x[2])))
        mean = training_rdd.map(lambda x: float(x[2])).mean()

        # Combine ground truth and prediction file, predict for users not in training data and Calculate RMSE
        gt_and_pred = ground_truth_index.union(predictions).reduceByKey(lambda x, y: (x, y))
        ratesAndPreds = gt_and_pred.map(lambda x: (x[0], (x[1], mean)) if type(x[1]) == float else x)

        with open(output_file_path, 'w') as fout:
            fout.write("user_id, business_id, prediction\n")
            for row in ratesAndPreds.collect():
                user_print = str(indexuserdict.get(row[0][0]))
                business_print = str(indexbusinessdict.get(row[0][1]))
                predicted_rating = str(row[1][1])
                fout.write(user_print + "," + business_print + "," + predicted_rating + "\n")
            fout.close()

    # User Based Collaborative Filtering
    if int(case_id) == 2:
        def get_business_avg(business):
            list_users = business_dict[business]
            mean_rating = 0.0
            for user in list_users:
                mean_rating += user_business_dict[(user, business)]
            mean_rating = mean_rating/len(list_users)
            return mean_rating


        def get_weight(user, neighbor):
            num = 0.0
            d1 = 0.0
            d2 = 0.0
            neighbor_data_dict = dict()
            user_data = user_data_dict[user]
            neighbor_data = user_data_dict[neighbor]
            for x in neighbor_data:
                neighbor_data_dict[x[0]] = x[1]

            for row in user_data:
                business = row[0]
                user_rating = row[1]
                # d2 += (user_rating ** 2)
                if business in neighbor_data_dict:
                    neighbor_rating = neighbor_data_dict[business]
                    d1 += (neighbor_rating * neighbor_rating)
                    d2 += (user_rating * user_rating)
                    num += (user_rating * neighbor_rating)

            den = math.sqrt(d1) * math.sqrt(d2)
            if num == 0.0 or den == 0.0:
                return -15.0
            else:
                return float(num / den)


        def get_neighbors(user, business):
            neighbor_dict = dict()
            if user not in user_data_dict:
                return neighbor_dict
            if business not in business_dict:
                return neighbor_dict

            neighbors = business_dict[business]
            for current_neighbor in neighbors:
                weight = get_weight(user, current_neighbor)
                if weight == -15.0:
                    return neighbor_dict
                else:
                    neighbor_dict[current_neighbor] = weight
            return neighbor_dict


        def predict(user, business, neighbor_dict):
            if bool(neighbor_dict) == False:
                if user not in user_data_dict:
                    return get_business_avg(business)
                else:
                    return mean_dict[user]

            num = 0.0
            den = 0.0
            list_users = business_dict[business]
            for neighbor in list_users:
                if neighbor in neighbor_dict:
                    if neighbor_dict[neighbor] > 0.0:
                        neighbor_rating = user_business_dict[(neighbor, business)]
                        neighbor_weight = neighbor_dict[neighbor]
                        num += (neighbor_rating * neighbor_weight)
                        den += abs(neighbor_weight)
            if (num == 0.0):
                predict = mean_dict[user]
            else:
                predict = mean_dict[user] + (num / den)
            return predict


        sc = SparkContext("local[*]", "User based recommendation system")

        # Store training and testing file in rdd without header
        all_rdd = sc.textFile(train_file_path)
        train_header = all_rdd.first()
        all_rdd = all_rdd.filter(lambda x: x != train_header).map(lambda x: x.split(','))
        all_rdd = all_rdd.map(lambda x: (x[0], (x[1], float(x[2]))))

        test_rdd = sc.textFile(test_file_path)
        test_header = test_rdd.first()
        test_file = test_rdd.filter(lambda x: x != test_header).map(lambda x: x.split(','))
        test_rdd = test_file.map(lambda x: (x[0], x[1])).sortByKey()

        training_rdd = all_rdd.map(lambda x: ((x[0], x[1][0]), float(x[1][1])))

        # Create a dictionary for mean of all users
        mean_dict = training_rdd.map(lambda x: (x[0][0], float(x[1]))).mapValues(lambda v: (v, 1)).reduceByKey(
            lambda a, b: (a[0] + b[0], a[1] + b[1])).mapValues(
            lambda v: v[0] / v[1]).collectAsMap()

        training_rdd = training_rdd.map(lambda x: (x[0][0], x[0][1], float(x[1]) - mean_dict[x[0][0]]))
        udd = training_rdd.map(lambda x: (x[0], (x[1], x[2])))

        # Create 3 dictionaries: 1. (User, business): rating 2. Business: user 3. User: (business, rating)
        user_business_dict = training_rdd.map(lambda x: ((x[0], x[1]), x[2])).sortByKey().collectAsMap()
        business_dict = training_rdd.map(lambda x: (x[1], x[0])).groupByKey().sortByKey().mapValues(list).collectAsMap()
        user_data_dict = udd.groupByKey().sortByKey().mapValues(list).collectAsMap()

        top_neighbors = test_rdd.map(lambda x: (x[0], x[1], get_neighbors(x[0], x[1])))
        prediction = top_neighbors.map(lambda x: ((x[0], x[1]), predict(x[0], x[1], x[2])))


        with open(output_file_path, 'w') as fout:
            fout.write("user_id, business_id, prediction\n")
            for row in prediction.collect():
                user_print = str(row[0][0])
                business_print = str(row[0][1])
                predicted_rating = str(row[1])
                fout.write(user_print + "," + business_print + "," + predicted_rating + "\n")
            fout.close()

    # Item Based Collaborative Filtering
    if int(case_id) == 3:

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


        def get_businessindex_dict(user):
            bdict = dict()
            c = 0
            for i in user:
                bdict[i] = c
                indexbusinessdict[c] = i
                c += 1
            return bdict


        def get_userindex_dict(business):
            udict = dict()
            c = 0
            for i in business:
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
            user = set(charmatrix[x[0]])
            business2 = set(charmatrix[x[1]])
            intersection = user & business2
            union = user | business2
            jsimilarity = len(intersection) / len(union)
            return x[0], x[1], jsimilarity


        def get_weight(business, neighbor):
            num = 0.0
            d1 = 0.0
            d2 = 0.0
            active_sum = 0.0
            neighbor_sum = 0.0
            total_corrated_business = 0
            neighbor_data_dict = dict()
            business_data = business_data_dict[business]
            neighbor_data = business_data_dict[neighbor]
            for x in neighbor_data:
                neighbor_data_dict[x[0]] = x[1]

            for row in business_data:
                current_user = row[0]
                business_rating = row[1]
                if current_user in neighbor_data_dict:
                    neighbor_rating = neighbor_data_dict[current_user]
                    active_sum += business_rating
                    neighbor_sum += neighbor_rating
                    total_corrated_business += 1

            if total_corrated_business == 0:
                return -15.0

            active_mean = float(active_sum / total_corrated_business)
            neighbor_mean = float(neighbor_sum / total_corrated_business)

            for row in business_data:
                current_user = row[0]
                business_rating = row[1]
                if current_user in neighbor_data_dict:
                    neighbor_rating = neighbor_data_dict[current_user]
                    num += ((business_rating - active_mean) * (neighbor_rating - neighbor_mean))
                    d1 += ((business_rating - active_mean) ** 2)
                    d2 += ((neighbor_rating - neighbor_mean) ** 2)

            den = math.sqrt(d1) * math.sqrt(d2)
            if num == 0.0 or den == 0.0:
                return -15.0
            else:
                return float(num / den)


        def get_neighbors(business, user):
            neighbor_dict = dict()
            if business not in business_data_dict:
                return neighbor_dict
            if user not in user_dict:
                return neighbor_dict

            neighbors = user_dict[user]

            for current_neighbor in neighbors:
                weight = get_weight(business, current_neighbor)
                if weight == -15.0:
                    neighbor_dict[business] = 0.0
                else:
                    neighbor_dict[current_neighbor] = weight
            return neighbor_dict


        def get_neighbor_LSH(business):
            neighbor_dict = dict()
            if (business not in b1_b2_dict) and (business not in b2_b1_dict):
                # Business does not have neighbors in LSH output
                return neighbor_dict

            if business not in business_data_dict:
                # Business never rated by any user (item cold start)
                return neighbor_dict

            other_bus = list()
            if business in b1_b2_dict:
                other_bus = other_bus + b1_b2_dict[business]
            if business in b2_b1_dict:
                other_bus = other_bus + b2_b1_dict[business]

            neighbors = list(set(other_bus))

            for current_neighbor in neighbors:
                if current_neighbor != business:
                    weight = get_weight(business, current_neighbor)
                    if weight != -15.0:
                        neighbor_dict[current_neighbor] = weight
                    else:
                        neighbor_dict[business] = 0.0

            return neighbor_dict


        def predict(business, user, neighbor_dict):
            if bool(neighbor_dict) == False:
                if business not in business_data_dict:
                    return 3
                else:
                    return mean_dict[business]
            if len(neighbor_dict) == 1 and business in neighbor_dict:
                    return mean_dict[business]

            num = 0.0
            den = 0.0
            list_business = user_dict[user]
            for neighbor in list_business:
                if neighbor in neighbor_dict:
                    if neighbor != business:
                        neighbor_rating = business_user_dict[(neighbor, user)]
                        neighbor_weight = neighbor_dict[neighbor]
                        num += ((neighbor_rating - mean_dict[neighbor]) * neighbor_weight)
                        den += abs(neighbor_weight)
            if (num == 0.0):
                predict = mean_dict[business]
            else:
                predict = mean_dict[business] + (num / den)
            return predict


        sc = SparkContext("local[*]", "Item based recommendation system with LSH")

        # Convert input file to text and remove header
        all_rdd = sc.textFile(train_file_path)
        train_header = all_rdd.first()
        all_rdd = all_rdd.filter(lambda x: x != train_header).map(lambda x: x.split(','))

        # Map all data into rdd and create dictionary of indices
        user = all_rdd.map(lambda x: x[0]).distinct().collect()
        user.sort()
        business = all_rdd.map(lambda x: x[1]).distinct().collect()
        business.sort()
        datalist = all_rdd.collect()

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

        similar_business = get_pair.map(jaccard_similarity).filter(lambda x: x[2] >= 0.5).sortBy(lambda x: x[1]).sortBy(lambda x: x[0]).map(lambda x: (indexbusinessdict.get(x[0]), indexbusinessdict.get(x[1]), x[2]))

        all_rdd = all_rdd.map(lambda x: (x[1], (x[0], float(x[2]))))

        test_rdd = sc.textFile(test_file_path)
        test_header = test_rdd.first()
        test_file = test_rdd.filter(lambda x: x != test_header).map(lambda x: x.split(','))
        test_rdd = test_file.map(lambda x: (x[1], x[0])).sortByKey()


        training_rdd = all_rdd.map(lambda x: ((x[0], x[1][0]), float(x[1][1])))


        # Create a dictionary for mean of all users
        mean_dict = training_rdd.map(lambda x: (x[0][0], float(x[1]))).mapValues(lambda v: (v, 1)).reduceByKey(
            lambda a, b: (a[0] + b[0], a[1] + b[1])).mapValues(
            lambda v: v[0] / v[1]).collectAsMap()

        training_rdd = training_rdd.map(lambda x: (x[0][0], x[0][1], float(x[1])))
        udd = training_rdd.map(lambda x: (x[0], (x[1], x[2])))


        # Create 5 dictionaries: 1. (User, business): rating 2. Business: user 3. User: (business, rating) 4. B1->B2 5. B2->B1 (Obtained from LSH)
        business_user_dict = training_rdd.map(lambda x: ((x[0], x[1]), x[2])).sortByKey().collectAsMap()
        user_dict = training_rdd.map(lambda x: (x[1], x[0])).groupByKey().sortByKey().mapValues(list).collectAsMap()
        business_data_dict = udd.groupByKey().sortByKey().mapValues(list).collectAsMap()
        b1_b2_dict = similar_business.map(lambda x: (x[0], x[1])).groupByKey().sortByKey().mapValues(list).collectAsMap()
        b2_b1_dict = similar_business.map(lambda x: (x[1], x[0])).groupByKey().sortByKey().mapValues(list).collectAsMap()

        # For item-item collaborative filtering with LSH:
        top_neighbors = test_rdd.map(lambda x: (x[0], x[1], get_neighbor_LSH(x[0])))

        # For item-item collaborative filtering without LSH:
        # top_neighbors = test_rdd.map(lambda x: (x[0], x[1], get_neighbors(x[0], x[1])))

        prediction = top_neighbors.map(lambda x: ((x[0], x[1]), predict(x[0], x[1], x[2])))

        with open(output_file_path, 'w') as fout:
            fout.write("user_id, business_id, prediction\n")
            for row in prediction.collect():
                user_print = str(row[0][0])
                business_print = str(row[0][1])
                predicted_rating = str(row[1])
                fout.write(user_print + "," + business_print + "," + predicted_rating + "\n")
            fout.close()

    else:
        exit(0)


# coding=UTF-8
import os            #os模块，操作系统的功能 
import os.path       #路径相关操作
import cv2           #导入openCV库
import glob       #支持通配符操作，*,?,[]这三个通配符，*代表0个或多个字符，?代表一个字符，[]匹配指定范围内的字符
import imutils     #提供一系列便捷功能进行基本的图像处理功能


CAPTCHA_IMAGE_FOLDER = "generated_captcha_images"
OUTPUT_FOLDER = "extracted_letter_images"


# Get a list of all the captcha images we need to process
captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
#os.path.join合并目录,将多个路径组合后返回
#glob.glob方法： 返回所有匹配的文件路径列表，该方法需要一个参数用来指定匹配的路径字符串
counts = {} #定义一个计数的字典

# loop over the image paths
for (i, captcha_image_file) in enumerate(captcha_image_files):
    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))  #.format逐步输出

    # Since the filename contains the captcha text (i.e. "2A2X.png" has the text "2A2X"),
    # grab the base filename as the text
    filename = os.path.basename(captcha_image_file) #os.path.basename(path)——返回文件名
    captcha_correct_text = os.path.splitext(filename)[0]  #用于对比的真实值

    # Load the image and convert it to grayscale
    image = cv2.imread(captcha_image_file)  #加载图像    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #cv2.cvtColor(input_image, flag)函数实现图片颜色空间的转换，flag 参数决定变换类型变成灰度图

    # Add some extra padding around the image
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)  #扩充图像，图像名，上下左右，BORDER_REFLICATE:直接用边界的颜色填充

    # threshold the image (convert it to pure black and white)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]#图像二值化 cv2.threshold(灰度图像，阈值，最大值，转换方式)

    # find the contours (continuous blobs of pixels) the image
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #提取图像轮廓 cv2.findContours(图像，提取规则。cv2.RETR_EXTERNAL：只找外轮廓，cv2.RETR_TREE：内外轮廓都找，输出轮廓内容格式。cv2.CHAIN_APPROX_SIMPLE：输出少量轮廓点。cv2.CHAIN_APPROX_NONE：输出大量轮廓点。)
      
    # Hack for compatibility with different OpenCV versions
    contours = contours[0] if imutils.is_cv2() else contours[1] #contours是一个轮廓的列表，0是随机的

    letter_image_regions = []  #空，用于保存提取出的四字字母

    # Now we can loop through each of the four contours and extract the letter
    # inside of each one
    for contour in contours:
        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour) #求包含轮廓的正方框（x，y）为矩形左上角的坐标，（w，h）是矩形的宽和高

        # Compare the width and height of the contour to detect letters that
        # are conjoined into one chunk
        if w / h > 1.25:
            # This contour is too wide to be a single letter!
            # Split it in half into two letter regions!
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            # This is a normal letter by itself
            letter_image_regions.append((x, y, w, h))

    # If we found more or less than 4 letters in the captcha, our letter extraction
    # didn't work correcly. Skip the image instead of saving bad training data!
    if len(letter_image_regions) != 4:
        continue

    # Sort the detected letter images based on the x coordinate to make sure
    # we are processing them from left-to-right so we match the right image
    # with the right letter
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])  #sorted 可以对所有可迭代的对象进行排序操作sorted(iterable[, cmp[, key[, reverse]]])
#key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序
#cmp -- 比较的函数，这个具有两个参数，参数的值都是从可迭代对象中取出，此函数必须遵守的规则为，大于则返回1，小于则返回-1，等于则返回0。
#reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）
    # Save out each letter as a single image
    for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box    #寻找字母坐标

        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]

        # Get the folder to save the image in
        save_path = os.path.join(OUTPUT_FOLDER, letter_text)

        # if the output directory does not exist, create it
        if not os.path.exists(save_path):    #os.makedirs() 方法用于递归创建目录
            os.makedirs(save_path)

        # write the letter image to a file
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)  #保存图像

        # increment the count for the current key
        counts[letter_text] = count + 1

# AIAP Technical Assessment — Decision Log

**Candidate name (as in NRIC):** Gautam Paramanantham

**Email (as used in your application):** gautamparam@gmail.com

---

## A note on this document

This decision log is the primary instrument by which we understand your thinking. The questions below cover the reasoning behind your work — from how you define the problem at the start to the decisions you made during the work itself. Do answer all five questions in your own words. 

You may use AI assistance freely on the technical deliverables (the EDA and the ML pipeline), but this Decision Log itself should be written by you — it is the record of your own thinking that we cross-check against your chat history.


---

## 1. Clarifying questions

What questions would you ask to better define and narrow the problem statement? For each question, briefly explain how the answer would meaningfully change your approach. 

Note: If it helps your decision-making, you may assume and list out the stakeholders' likely answers.


**Your answer:**

1. What exactly would you like us to build for you? I see 2 likely answers:

    1a. Build us a module that will give us for any booking, at the time of the booking, how likely it is that the customer will cancel it. We will take that probability, multiply it by the booking value, use that as a severity gauge and execute a triage plan if needed.
    1b. Build us a module that will dynamically set the price of rooms daily/regularly in a way that will maximize revenue. 

    We went with 1a for this assignment because we assumed the hotel just wanted to do whatever that would recover the most lost revenue. If they had a preference for 1b, that would change the model that we build out for them. 

2. What is more important - pricing the rooms correctly, or minimizing cancellations without deposits? Likely answer:

    2a. Whatever recovers more revenue for us, without increasing cost or manpower requirements.

    Again, like in 1, if there were any 'hidden' requirements besides just recovering lost revenue, this would change the direction we took the assignment in.
    
3. Are there any non-financial/intangible considerations or guidelines that you want us to keep in mind as you're building out this module? Likely answer:

    3a. Treat our repeat customers well, give them an excellent experience. If we're doing anything on differentiated pricing down the road, we want to try our best to give our repeat customers consistent and reasonable rates.

    Ditto as in 1 and 2, we want to clarify that our key assumption of recovering revenue is the key goal. If they want to maintain a great experience for repeat customers (e.g. business travellers they have company contracts with), we will want to remove these list of customers from our module, and make sure that these customers do not get affected by the module when they're going through the booking process.



---

## 2. Defining the Problem Statement

Restate, in your own words, the refined problem you decided to solve. List your key assumptions. Briefly note what other framings you considered, and what you deliberately left out or scoped down, and why.


**Your answer:** 
- **Problem statement:** Travelers' Inn loses a lot of revenue from cancelled bookings. Build a model to predict the expected cancellation value [P(cancel) x value of booking] of a booking at the time of booking, so that Travelers Inn can implement measures to triage bookings with large expected cancellation values. The hotel already has the value of the booking at the time of booking, so we just need to predict the probability of the booking being cancelled.
- **Other framings I considered:** I started the assignment with 5 options for the problem statement, some of them involved a conditional probability (P cancellation given the seasonality), and also an added regression model that would give the forgone-revenue from mispriced ADRs on top of just the probability of cancellation. But I decided to keep it simple and go with just the P(cancellation) for the model, and then multiply that by booking value which we'll anyway have at the time of booking.
- **What I deliberately scoped down:** At the start, I was considering weighting the set so that the model would prioritize bookings that were more likely to result in unrecoverable revenue. But I decided to simplify the model as doing this I thought would overcomplicate the model. I instead added a recoverability-weighted metric for the evaluation as a sanity check to see how we actually do on revenue that is really lost. 

**Key assumptions:**
1. The revenue lost from booking cancellations is significantly more than the revenue lost from mis-pricing rooms. This was validated by deriving the room capacities for each of the 2 hotels, and deriving the revenue lost from unrecoverable cancellations (i.e. occupancy doesn't recover to pre-cancellation levels), and comparing that against estimated revenue loss from mispriced ADRs.
2. Inferred capacity for the airport branch is 185 rooms, and city branch is 225 rooms - please see below sections for details on how I derived this.
3. We use margins and revenue synonymously in this assignment. This is because usually for hotels, most of the costs are fixed. So if we get additional revenue from filling otherwise empty rooms, most of it less aggregator and payment gateway fees goes straight to the bottom line.





---

## 3. Key decisions during Solution Development

Walk through three key decisions you made during Solution Development. For each: what options did you consider, what did you choose, and why? These can be technical (modelling choices, feature handling, evaluation metrics) or about the work itself (what to prioritise, what to drop, how to spend your time).


**Your answer:**

I pretty much used AI extensively and freely throughout the assignment, except when filling this document. I used AI to guide my decisions and bounce ideas, but I always made sure that I understood and made the final decisions myself. As mentioned below, there were many times I overrode the AI's decision, the below are the major examples

1. After reading the assignment and before I started work on the problem statement, I queried the database, made some notes, and formed some initial hypotheses on what the problem may be. As I dug into the dataset and wrote my thoughts down, it became apparent to me that we were trying to minimize revenue loss, and the two main ways to do it were to reduce cancellation rates and improve the way we price rooms. I decided to come up with a first cut of 5 potential problem statements around this key insight, before doing the EDA. The plan was to start with some hyptheses, do the EDA, and then after the EDA, revisit, re-evaluate the model options and refine it based on my key findings. I did this pre-EDA hypotheses and post-EDA revisit so we could be more focused with the exploration, and use that to refine our framing of the problem.
2. I also got Claude to derive the total number of rooms in the airport branch (185 rooms), and city branch (225), and use this to estimate how much money we lost from cancellations vs mispriced ADRs. This was a rough gauge, but it allowed me to at least get an idea of what potential lost revenue capture we'd be looking at for each, and prioritize one.I got Claude to derive the total number of rooms by looking at where we maxed out in terms of occupancy for each hotel. This allowed me to properly gauge later on what exactly we were losing on - e.g. we're under-pricing if we're constantly at 100% occupancy at a specific ADR and season, or cancellation is not a problem if we're still at 99% occupancy after cancellations. Through this, I found that losses from unrecoverable cancellations were significantly higher than the estimated losses from mis-priced ADRs, it allowed me to make an initial guess that focusing on predicting cancellations was the way to go. This was not yet a final decision, the plan was to go through the EDA and then see if this was indeed the case, and refine the problem statement if needed.  I thought this hypotheses-driven approach, coupled with a post-EDA refinement, would land us a more relevant problem statement. 
3. The most important override I made over Claude's recommendation was to switch the model over from a hist_gbm model to a much simpler logistic regression model. I did this because the logistic regression model did almost as well as this model on all the key metrics except the PR-AUC. The details of the comparison and the delta between the models is in the readme. I thought that going with a simpler model that is easier to interpret and retrain was the more 'mature' choice to make.   



---

## 4. Use of the AI assistant

Where did you use the AI assistant in this work? Give three specific examples of something the assistant suggested that you changed, rejected, or significantly modified — and explain your reasoning.


**Your answer:**

1. During the initial exploratory data analysis, all cancellations were bunched into one analysis, and the key conclusion drawn was that many cancellations are not last minute like our project point of contact said. I thought this was the wrong approach, and a better way to approach this would be to take a closer look at last minute cancellations instead of just blanketing all cancellations. This is because I thought that cancellations with lots of heads up could be recovered with a rebooking, and hence don't contribute as much to revenue loss. After some digging, instead of going the way of weighting last-minute cancellations, I decided to just add in an evaluation metric that assesses how the model does against unrecovered revenue. This would be a good sanity check.
2. After we were done with the EDA, Claude suggested that we go with 'P(cancel) × P(lost | date) × value' as the model. I thought that this would be a complicated model to build, evaluate and retrain. Because of this, I pushed back and asked Claude to estimate the lost revenue recaptured from running this model vs just a simple P(cancel) model. After some conversation, we added an intermediate model, 'P(cancel) x value' to the mix, and I had Claude evaluate all 3 models and see which model would recapture most of the losses. The middle model did the best, and the P(cancel) model, despite being the simplest, didn't do well enough to justify going with it. So in the end, I ended up rejecting Claude's proposed model, and went with a simpler 'P(cancel) x value' model instead. 
3. After the EDA, when I moved on to building the pipeline, Claude consistently tried to one-shot the entire pipeline at every message. I had to explicitly ask Claude to break down the steps into a plan, and I had to add in checks in the plan, such as unit tests, and before and after visual comparisons for the feature processing, so I could eyeball eveything and make sure it's fine. By doing this, I caught other critical issues, like Claude not stating the how the training and testing data was split, wrongly stating that it weighted last-minute cancellations more heavily, when in fact didn't - I ultimately decided against doing this myself since passing a sample_weight = realized loss fit would significantly complicate the model. It also wrongly stated that it had run these numbers against a simple dummy baseline, but the dummy check was not properly set up.




---

## 5. Next Steps

If you had one more week to continue this project, what would you do next, and why? What signals from your current work make those the right next steps?


**Your answer:**

1. Focus on how we're going to implement this. This model only gives us information, what we do with it is just as important. My guess is that after identifying high-expected-loss-value bookings, we will try to collect non-refundable deposits from them during the time of transactions. If we end up doing this, we should track the drop in conversion rates and occupancy rates from doing this, and compare this loss against the loss from cancellations. The ask to improve the profit margins (and hence revenue) - we should implement and make sure it's actually doing this.
2. Try weighting the model to catch unrecoverable revenue like I initially planned. We could do this by passing a sample_weight feature that equals the actual realized loss into the fit. Compare this model against the current in production side-by-side.
3. Look into a pricing regression model. This will address our point of contact's other ask, and address the 2nd largest source of revenue leak - mispriced ADRs. We currently have ~32,000 empty room nights per year - I would imagine that dropping prices to fill up these room nights would be much better for our bottom-line than leaving these empty.




---